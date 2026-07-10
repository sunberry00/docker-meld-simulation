"""
run_experiment.py — one full FL simulation cycle, end to end.

Every phase of the experiment is explicit and separately runnable; this script
just chains them. The phases map 1:1 to the project layout:

    Phase 1  pipeline/02_make_clinic_dumps.py   dump -> 5 shifted clinic dumps
    Phase 2  docker-compose.fl.yaml up          5x (Postgres i2b2 + MELD orchestrator)
    Phase 3  pipeline/03_load_clinics.py        clinic dumps -> clinic databases
    Phase 4  wait for orchestrator APIs
    Phase 5  fl_server.py                       N rounds of FedAvg over 5 sites
    Phase 6  collect results into experiments/<scenario>/
    Phase 7  docker-compose.fl.yaml down -v     destroy everything

Phase 0 (pipeline/01_export_dump.py — source DWH -> immutable parquet dump)
is run ONCE beforehand, not per experiment, because the dump never changes.

Usage:
    python run_experiment.py --config config/experiment.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))

# The numbered pipeline files are imported by module name (importlib handles
# names that start with digits).
import importlib

_sim = importlib.import_module("02_make_clinic_dumps")
_load = importlib.import_module("03_load_clinics")


# ---------------------------------------------------------------- helpers

def _run(cmd: str, check: bool = True) -> None:
    """Run a shell command, streaming output."""
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=check)


def _wait_for_apis(urls: list[str], timeout: int = 120) -> None:
    """Poll each orchestrator's OpenAPI page until all respond."""
    import requests

    t0 = time.time()
    pending = set(urls)
    while pending and time.time() - t0 < timeout:
        for url in list(pending):
            try:
                if requests.get(f"{url}/health", timeout=3).status_code == 200:
                    pending.discard(url)
                    print(f"    {url} ready")
            except Exception:
                pass
        if pending:
            time.sleep(3)
    if pending:
        raise TimeoutError(f"Orchestrators not ready after {timeout}s: {sorted(pending)}")


# ---------------------------------------------------------------- phases

def run_experiment(config_path: str) -> None:
    import pandas as pd

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    scenario = cfg["scenario"]
    seed = cfg.get("seed", 42)
    n_sites = cfg.get("n_sites", 5)
    out_dir = cfg.get("out_dir", f"experiments/{scenario}")
    compose = cfg.get("compose_file", "docker-compose.fl.yaml")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Phase 1: shifted clinic dumps -----------------------------------
    print("\n=== Phase 1: Generate shifted clinic dumps ===")
    df = pd.read_parquet(cfg["dump_path"])
    if "admitted" not in df.columns and "verbleib" in df.columns:
        df["admitted"] = df["verbleib"].astype(str).str.startswith("Aufnahme").astype(int)
    with open(cfg["simulation_config"]) as f:
        sim_cfg = yaml.safe_load(f)
    sim_out = os.path.join(out_dir, "clinic_dumps")
    manifest = _sim.run_scenario(
        df, scenario, sim_cfg["scenarios"][scenario], seed, sim_out,
        holdout_frac=cfg.get("holdout_frac", 0.15),
        train_frac=cfg.get("train_frac", 0.70),
        val_frac=cfg.get("val_frac", 0.15),
    )
    print(f"  {manifest['n_sites']} clinic dumps -> {sim_out}/")

    # ---- Phase 2: spin up clinics ----------------------------------------
    print("\n=== Phase 2: Spin up 5 clinics (Postgres + MELD orchestrator) ===")
    _run(f"docker compose -f {compose} up -d")
    time.sleep(10)  # give the healthchecked DBs a head start

    dwh_base = cfg.get("dwh_base_port", 5440)
    api_base = cfg.get("api_base_port", 8100)

    try:
        # ---- Phase 3: load clinic dumps into clinic DBs ------------------
        print("\n=== Phase 3: Load clinic dumps into clinic databases ===")
        for i in range(n_sites):
            parquet = os.path.join(sim_out, f"site_{i}_train.parquet")
            target = f"postgresql://i2b2crcdata:demouser@localhost:{dwh_base + i}/i2b2"
            print(f"\n  site_{i} (train split) -> port {dwh_base + i}")
            _load.load_site(parquet, cfg["source_db"], target, clear=True)

        # ---- Phase 4: wait for orchestrator APIs -------------------------
        print("\n=== Phase 4: Wait for MELD orchestrator APIs ===")
        site_urls = [f"http://localhost:{api_base + i}" for i in range(n_sites)]
        _wait_for_apis(site_urls)

        # ---- Phase 5: federated training ----------------------------------
        print("\n=== Phase 5: Federated training ===")
        fl_out = os.path.join(out_dir, "fl_results")
        _run(
            "python fl_server.py "
            f"--sites {' '.join(site_urls)} "
            f"--rounds {cfg.get('fl_rounds', 10)} "
            f"--backbone {cfg['initial_backbone']} "
            f"--out {fl_out}"
        )

        # ---- Phase 6: collect results --------------------------------------
        print("\n=== Phase 6: Collect results ===")
        summary = {
            "scenario": scenario,
            "seed": seed,
            "n_sites": n_sites,
            "fl_rounds": cfg.get("fl_rounds", 10),
            "clinic_manifest": manifest,
            "fl_results_dir": fl_out,
        }
        with open(os.path.join(out_dir, "experiment_result.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  results -> {out_dir}/experiment_result.json")

    finally:
        # ---- Phase 7: tear down (always) -----------------------------------
        print("\n=== Phase 7: Tear down all clinics ===")
        _run(f"docker compose -f {compose} down -v", check=False)

    print("\nExperiment complete. Only the results remain.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one full FL simulation experiment")
    ap.add_argument("--config", default="config/experiment.yaml")
    args = ap.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
