"""
Training runner for the MELD orchestrator.

Mirrors ModelEnvironment.inference but uses a **shared Docker volume** instead of
tar-based file copy. The volume is mounted at ``/input`` (read) and ``/output``
(write) inside the runtime container. The orchestrator writes training inputs
(CSV, contract, backbone weights, optional adapter weights) to the host side of
the volume before starting the container, and reads the results (updated backbone,
adapter, metadata) from the output side after the container exits.

The container image is the same as for inference; the ``MELD_MODE=train``
environment variable tells ``entry.py`` which code path to run.
"""
from __future__ import annotations

import datetime
import io
import json
import os
import shutil
import tarfile

import docker
import pandas as pd
import yaml
from docker.models.containers import Container

from ModelEnvironment.docker_runtime import (
    start_container,
    stop_container,
    wait_for_container,
    destroy_container,
    ensure_image_exists,
)
from ModelEnvironment.job_context import JobContext, JobStatus
from Logger.logger import get_meld_logger

logger = get_meld_logger()

_docker = docker.from_env()


# ------------------------------------------------------------------ helpers

def _prepare_shared_volume(
    job_context: JobContext,
    input_data: pd.DataFrame,
    backbone_path: str | None,
    adapter_path: str | None,
) -> str:
    """Write all training inputs into the job's input folder (host side).

    Returns the absolute host path that will be bind-mounted as ``/input``
    inside the container.
    """
    host_input = job_context.input_data_path  # <root>/jobs/<job_id>/input

    input_data.to_csv(os.path.join(host_input, "input.csv"), index=False)

    with open(os.path.join(host_input, "contract.yaml"), "w") as f:
        yaml.dump(job_context.contract, f)

    if backbone_path and os.path.isfile(backbone_path):
        shutil.copy2(backbone_path, os.path.join(host_input, "backbone.pt"))

    if adapter_path and os.path.isfile(adapter_path):
        shutil.copy2(adapter_path, os.path.join(host_input, "adapter.pt"))

    job_context.logger.info(
        f"Prepared shared volume input at {host_input} "
        f"(backbone={'yes' if backbone_path else 'no'}, "
        f"adapter={'yes' if adapter_path else 'no'})"
    )
    return host_input


def _create_training_container(
    image: str,
    job_context: JobContext,
    host_input: str,
    host_output: str,
) -> Container:
    """Create a container with shared-volume bind mounts and MELD_MODE=train."""
    env = dict(job_context.contract["runtime"].get("environment_variables", {}))
    env["MELD_MODE"] = "train"

    container = _docker.containers.create(
        image,
        environment=env,
        volumes={
            host_input:  {"bind": "/input",  "mode": "ro"},
            host_output: {"bind": "/output", "mode": "rw"},
        },
    )
    job_context.log_event(
        f"Created training container {container.id}",
        JobStatus.CREATED,
        image=image,
    )
    return container


def _read_training_output(job_context: JobContext) -> dict:
    """Read metadata.json from the output folder after training completes."""
    meta_path = os.path.join(job_context.output_data_path, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    job_context.logger.warning("No metadata.json found in training output")
    return {}


# ------------------------------------------------------------------ public

def run_training(
    input_data: pd.DataFrame,
    job_context: JobContext,
    backbone_path: str | None = None,
    adapter_path: str | None = None,
) -> dict:
    """Run one local training round inside a runtime container.

    Parameters
    ----------
    input_data : pd.DataFrame
        Training data (features + label column) from the DWH query.
    job_context : JobContext
        The job context for this run (folders, logging, contract).
    backbone_path : str, optional
        Path to the current global backbone weights (``backbone.pt``).
    adapter_path : str, optional
        Path to the site-local LoRA adapter from the previous round.

    Returns
    -------
    dict
        The contents of ``/output/metadata.json`` written by the container
        (typically ``{"n_samples": int, "train_loss": float}``).  The updated
        ``backbone.pt`` and ``adapter.pt`` are in ``job_context.output_data_path``.
    """
    job_context.log_event("Preparing training", JobStatus.PREPARING)
    image = job_context.image_ref

    ensure_image_exists(job_context)

    host_input = _prepare_shared_volume(
        job_context, input_data, backbone_path, adapter_path
    )
    host_output = job_context.output_data_path

    container = None
    try:
        container = _create_training_container(
            image, job_context, host_input, host_output
        )

        t0 = datetime.datetime.now()
        start_container(container, job_context)

        job_context.log_event("Training running", JobStatus.RUNNING)
        wait_for_container(container, job_context)

        stop_container(container, job_context)
        elapsed = (datetime.datetime.now() - t0).total_seconds()
        job_context.logger.info(f"Training completed in {elapsed:.1f}s")

        meta = _read_training_output(job_context)
        job_context.log_event("Training completed", JobStatus.SUCCESS)
        return meta

    except Exception as e:
        job_context.logger.exception(f"Training failed: {e}")
        job_context.log_event("Training failed", JobStatus.FAILED, error=str(e))
        return {}
    finally:
        if container is not None:
            destroy_container(container, job_context)
