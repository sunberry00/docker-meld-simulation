"""
MELD orchestrator REST API — Flask, raw bytes for weights.

    POST /start         Body = backbone.pt bytes (or empty).
                        Returns JSON: {"job_id": "..."}

    GET  /status/<id>   Returns JSON: {"state": "...", ...}

    GET  /results/<id>  Returns trained backbone.pt as raw bytes.

    GET  /health        Liveness check.
"""
import json
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, Response

app = Flask(__name__)

CONTRACT_PATH = os.environ.get("MELD_CONTRACT_FILE", "/resources/contract.yaml")
ADAPTER_STORE = os.environ.get("MELD_ADAPTER_STORE", "/adapter_store")


# ---------------------------------------------------------------- job store
@dataclass
class TrainJob:
    job_id: str
    state: str = "Started"
    output_dir: str = ""
    metadata: dict = field(default_factory=dict)
    error: str = ""

_jobs: dict[str, TrainJob] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------- worker
def _resolve_params(scope: dict) -> dict:
    """Turn the contract's temporal_scope into :start/:end query parameters."""
    if not scope or scope.get("type") == "none":
        return {"start": "1900-01-01", "end": "2099-12-31"}
    if scope.get("type") == "absolute":
        return {"start": scope["start"], "end": scope["end"]}
    anchor = scope.get("anchor")
    end = datetime.fromisoformat(anchor) if anchor else datetime.now()
    value = scope.get("value", "-P1Y")
    dur = value.lstrip("-")
    days = 365
    if "Y" in dur:
        days = int(dur.replace("P", "").replace("Y", "")) * 365
    elif "M" in dur:
        days = int(dur.replace("P", "").replace("M", "")) * 30
    elif "D" in dur:
        days = int(dur.replace("P", "").replace("D", ""))
    start = end - timedelta(days=days)
    return {"start": start.isoformat(), "end": end.isoformat()}


def _train_worker(job: TrainJob, backbone_bytes: bytes | None) -> None:
    """Background thread: use MELD's own JobContext to query DWH, then run
    the training container with shared volumes."""
    try:
        job.state = "Running"

        # Use MELD's JobContext — it loads the contract, creates job folders,
        # sets up logging, and provides .contract for execute_query.
        from ModelEnvironment.job_context import JobContext
        from InternalDataLoader import execute_query
        import docker

        job_context = JobContext(CONTRACT_PATH)
        job.output_dir = job_context.output_data_path

        # Override the auto-generated job_id with ours for consistency.
        # (JobContext already created folders, that's fine.)

        # 1. Query the clinic DWH using MELD's standard data path.
        scope = job_context.contract.get("input_schema", {}).get("temporal_scope", {})
        params = _resolve_params(scope)
        df = execute_query(job_context, params)
        job_context.logger.info(f"Query returned {len(df)} rows for training")

        # 2. Write training inputs to the job's input folder.
        input_dir = job_context.input_data_path
        output_dir = job_context.output_data_path

        df.to_csv(os.path.join(input_dir, "input.csv"), index=False)

        if backbone_bytes:
            with open(os.path.join(input_dir, "backbone.pt"), "wb") as f:
                f.write(backbone_bytes)

        adapter_src = os.path.join(ADAPTER_STORE, "adapter.pt")
        if os.path.isfile(adapter_src):
            shutil.copy2(adapter_src, os.path.join(input_dir, "adapter.pt"))

        import yaml
        with open(os.path.join(input_dir, "contract.yaml"), "w") as f:
            yaml.dump(job_context.contract, f)

        # 3. Run the training container.
        #
        # We do NOT bind-mount input_dir/output_dir: the orchestrator talks to the
        # HOST docker daemon (docker-out-of-docker), so a bind-mount source path
        # would be resolved on the host, where /jobs/<id> does not exist. Instead
        # we stream files in/out via the Docker API with put_archive/get_archive,
        # exactly like upstream MELD's inference path.
        import io
        import tarfile

        client = docker.from_env()
        image = job_context.image_ref
        env = dict(job_context.contract.get("runtime", {}).get("environment_variables", {}))
        env["MELD_MODE"] = "train"

        # Create (but don't start) the container so we can copy files in first.
        container = client.containers.create(image, environment=env)
        try:
            # Step 1: create /input and /output as directories inside the container
            # by dropping DIRTYPE tar entries at "/". This mirrors upstream MELD and
            # does not depend on the image having pre-created the folders.
            dir_buf = io.BytesIO()
            with tarfile.open(fileobj=dir_buf, mode="w") as tar:
                for d in ("input", "output"):
                    info = tarfile.TarInfo(d)
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o755
                    tar.addfile(info)
            dir_buf.seek(0)
            container.put_archive("/", dir_buf.getvalue())

            # Step 2: pack everything in input_dir into a tar and drop it at /input.
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                tar.add(input_dir, arcname="")
            buf.seek(0)
            container.put_archive("/input", buf.getvalue())

            container.start()
            result = container.wait()
            logs = container.logs().decode("utf-8", errors="replace")
            job_context.logger.info(f"Container logs:\n{logs}")
            # Persist container logs next to the job outputs so they survive the
            # container's destruction and can be fetched via GET /logs/{job_id}.
            with open(os.path.join(output_dir, "container_logs.txt"), "w") as lf:
                lf.write(logs)

            if result["StatusCode"] != 0:
                raise RuntimeError(f"Container exited with code {result['StatusCode']}:\n{logs}")

            # Pull /output/ back out of the container into output_dir.
            stream, _ = container.get_archive("/output")
            out_buf = io.BytesIO()
            for chunk in stream:
                out_buf.write(chunk)
            out_buf.seek(0)
            with tarfile.open(fileobj=out_buf, mode="r") as tar:
                for member in tar.getmembers():
                    # members come prefixed with "output/"; strip it
                    name = member.name.split("/", 1)[-1] if "/" in member.name else member.name
                    if not name or member.isdir():
                        continue
                    f = tar.extractfile(member)
                    if f is not None:
                        with open(os.path.join(output_dir, name), "wb") as out:
                            out.write(f.read())
        finally:
            container.remove(force=True)

        # 4. Read metadata.
        meta_path = os.path.join(output_dir, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                job.metadata = json.load(f)

        # 5. Persist adapter for next round.
        adapter_out = os.path.join(output_dir, "adapter.pt")
        if os.path.isfile(adapter_out):
            os.makedirs(ADAPTER_STORE, exist_ok=True)
            shutil.copy2(adapter_out, os.path.join(ADAPTER_STORE, "adapter.pt"))

        job.state = "Completed"
        print(f"[job {job.job_id}] Completed: {job.metadata}")

    except Exception as e:
        job.state = "Failed"
        job.error = str(e)
        import traceback
        print(f"[job {job.job_id}] FAILED: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------- routes
@app.route("/start", methods=["POST"])
def start_training():
    backbone_bytes = request.get_data() or None
    job_id = str(uuid.uuid4())[:8]
    job = TrainJob(job_id=job_id)
    with _lock:
        _jobs[job_id] = job

    thread = threading.Thread(target=_train_worker, args=(job, backbone_bytes), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id}), 202


@app.route("/status/<job_id>", methods=["GET"])
def get_status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404
    resp = {"job_id": job.job_id, "state": job.state}
    if job.state == "Completed":
        resp["metadata"] = job.metadata
        resp["backbone_ready"] = os.path.isfile(os.path.join(job.output_dir, "backbone.pt"))
    if job.state == "Failed":
        resp["error"] = job.error
    return jsonify(resp)


@app.route("/results/<job_id>", methods=["GET"])
def get_results(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None or job.state != "Completed":
        return jsonify({"error": "not ready"}), 404
    path = os.path.join(job.output_dir, "backbone.pt")
    if not os.path.isfile(path):
        return jsonify({"error": "backbone.pt not found"}), 404
    with open(path, "rb") as f:
        data = f.read()
    return Response(data, mimetype="application/octet-stream")


@app.route("/logs/<job_id>", methods=["GET"])
def get_logs(job_id: str):
    """Return the runtime container's logs for a job (also available for FAILED
    jobs — that is exactly when they matter most)."""
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job_id"}), 404
    path = os.path.join(job.output_dir, "container_logs.txt") if job.output_dir else ""
    if not path or not os.path.isfile(path):
        # No container logs (e.g. failure before the container started) —
        # return the job's error message instead so the caller still gets context.
        return Response(f"[no container logs]\njob state: {job.state}\nerror: {job.error}",
                        mimetype="text/plain")
    with open(path) as f:
        return Response(f.read(), mimetype="text/plain")


@app.route("/adapter", methods=["GET", "DELETE"])
def get_adapter():
    """GET: return this site's current LoRA adapter (from the persistent store).
    DELETE: reset the store — REQUIRED between benchmark runs of different
    backbones, because the volume outlives a single FL run and a stale adapter
    from another architecture would otherwise be silently loaded in round 1.

    NOTE: both verbs exist for the SIMULATION's host-side harness only. In a
    real deployment the adapter never leaves (nor is remotely wiped from) the site.
    """
    path = os.path.join(ADAPTER_STORE, "adapter.pt")
    if request.method == "DELETE":
        if os.path.isfile(path):
            os.remove(path)
            return jsonify({"status": "adapter removed"})
        return jsonify({"status": "no adapter stored"})
    if not os.path.isfile(path):
        return jsonify({"error": "no adapter stored"}), 404
    with open(path, "rb") as f:
        return Response(f.read(), mimetype="application/octet-stream")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("MELD_API_PORT", "8000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
