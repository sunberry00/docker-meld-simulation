"""
fl_server.py — Federated Learning server.

Coordinates N MELD orchestrator sites via their REST API. Weights are sent
and received as raw bytes in the request/response body (per Alexander's
guidance: "agnostisch mit den Gewichten als raw string im Body").

    POST /start     body = backbone.pt bytes    -> {"job_id": "..."}
    GET  /status/X                              -> {"state": "Completed", ...}
    GET  /results/X                             -> raw backbone.pt bytes

Each round:
    1. POST current backbone to every site -> collect job_ids.
    2. Poll /status until all Completed.
    3. GET /results -> download backbone bytes.
    4. FedAvg (weighted by n_samples) -> new global backbone.
    5. Save checkpoint, repeat.

Usage:
    python fl_server.py \
        --sites http://localhost:8100 http://localhost:8101 ... \
        --rounds 10 --backbone initial_backbone.pt --out fl_results/
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict

import requests
import torch


# ---------------------------------------------------------------- FedAvg

def fed_avg(updates: list[dict]) -> OrderedDict:
    """Weighted average of backbone state dicts by n_samples."""
    total = sum(u["n_samples"] for u in updates)
    if total == 0:
        return updates[0]["state"]
    avg = OrderedDict()
    for key in updates[0]["state"]:
        avg[key] = sum(
            u["state"][key].float() * (u["n_samples"] / total) for u in updates
        )
    return avg


# ---------------------------------------------------------------- round

def _dispatch(sites: list[str], backbone_path: str) -> dict[str, str]:
    """POST raw backbone bytes to every site, return {url: job_id}."""
    with open(backbone_path, "rb") as f:
        backbone_bytes = f.read()

    jobs = {}
    for url in sites:
        resp = requests.post(
            f"{url}/start",
            data=backbone_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30,
        )
        resp.raise_for_status()
        jobs[url] = resp.json()["job_id"]
        print(f"  dispatched to {url} -> job {jobs[url]}")
    return jobs


def _poll_until_done(jobs: dict[str, str], poll_interval: float = 5.0,
                     timeout: float = 600.0) -> dict[str, dict]:
    """Poll all sites until every job is Completed or Failed."""
    results: dict[str, dict] = {}
    pending = dict(jobs)
    t0 = time.time()

    while pending:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Round timed out after {timeout}s, pending: {list(pending)}")
        for url, job_id in list(pending.items()):
            try:
                resp = requests.get(f"{url}/status/{job_id}", timeout=10)
                data = resp.json()
                state = data.get("state", "")
                if state == "Completed":
                    results[url] = data
                    del pending[url]
                    n = data.get("metadata", {}).get("n_samples", "?")
                    print(f"  {url} completed (n_samples={n})")
                elif state == "Failed":
                    print(f"  {url} FAILED: {data.get('error', '?')}")
                    del pending[url]
            except Exception as e:
                pass  # retry next poll
        if pending:
            time.sleep(poll_interval)
    return results


def _collect_backbones(results: dict[str, dict], round_dir: str) -> list[dict]:
    """Download backbone bytes from each completed site."""
    updates = []
    for i, (url, data) in enumerate(results.items()):
        job_id = data["job_id"]
        resp = requests.get(f"{url}/results/{job_id}", timeout=30)
        resp.raise_for_status()

        local_path = os.path.join(round_dir, f"backbone_site{i}.pt")
        with open(local_path, "wb") as f:
            f.write(resp.content)

        state = torch.load(local_path, map_location="cpu", weights_only=True)
        n = data.get("metadata", {}).get("n_samples", 1)
        updates.append({"state": state, "n_samples": n})
    return updates


# ---------------------------------------------------------------- main loop

def run_fl(sites: list[str], n_rounds: int, backbone_path: str,
           out_dir: str, poll_interval: float = 5.0) -> None:
    os.makedirs(out_dir, exist_ok=True)
    global_path = os.path.join(out_dir, "global_backbone.pt")

    if backbone_path != global_path:
        with open(backbone_path, "rb") as s, open(global_path, "wb") as d:
            d.write(s.read())

    for r in range(n_rounds):
        print(f"\n{'='*60}\nRound {r+1}/{n_rounds}\n{'='*60}")
        round_dir = os.path.join(out_dir, f"round_{r+1}")
        os.makedirs(round_dir, exist_ok=True)

        jobs = _dispatch(sites, global_path)
        results = _poll_until_done(jobs, poll_interval)

        if not results:
            print("  no sites completed — skipping aggregation")
            continue

        updates = _collect_backbones(results, round_dir)
        new_state = fed_avg(updates)

        torch.save(new_state, global_path)
        torch.save(new_state, os.path.join(round_dir, "global_backbone.pt"))

        log = {"round": r + 1, "n_sites": len(results),
               "n_samples": [u["n_samples"] for u in updates]}
        with open(os.path.join(round_dir, "round_log.json"), "w") as f:
            json.dump(log, f, indent=2)
        print(f"  aggregated {len(updates)} sites -> new global backbone")

    print(f"\nFL complete. Final backbone: {global_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="FL server for MELD")
    ap.add_argument("--sites", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", default="fl_results/")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    args = ap.parse_args()
    run_fl(args.sites, args.rounds, args.backbone, args.out, args.poll_interval)


if __name__ == "__main__":
    main()
