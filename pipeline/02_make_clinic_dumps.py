"""
simulate_clinics.py — controlled clinic heterogeneity for FL experiments.

Takes the immutable parquet dump (from export_dwh_data.py) and produces N
clinic-specific subsets. Each subset is a list of encounter_nums — the actual
data is never modified, only *sampled differently*. This is a deliberate design
choice from the assignment: "Der ursprüngliche Dump darf nicht manuell
verändert werden; alle Änderungen müssen über Skripte nachvollziehbar erzeugt
werden."

Four levels of heterogeneity, applied independently then combined:

    Level 1 — Case volume       Different n per clinic.
    Level 2 — Age distribution  Stratified sampling by age bands.
    Level 3 — Gender mix        Stratified sampling by gender.
    Level 4 — MTS triage mix    Stratified sampling by triage score.

Usage:
    python simulate_clinics.py \\
        --dump data/raw/dwh_dump_seed_fixed.parquet \\
        --config config/simulation.yaml \\
        --out data/simulated/

Outputs per scenario:
    data/simulated/<scenario>/site_0.parquet ... site_N.parquet
    data/simulated/<scenario>/manifest.json   (reproducibility record)
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml


# ================================================================ core

TIME_COL = "admission_ts"


def carve_global_holdout(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off a global held-out test set BEFORE clinics are sampled.

    These rows go into NO clinic DWH. The final central / FedAvg / FedAvg+LoRA
    models are evaluated on this set as a single shared benchmark, so no clinic
    ever trained on it. Chronological: the latest `frac` of encounters are held out.
    """
    if frac <= 0:
        return df.reset_index(drop=True), df.head(0)
    d = df.sort_values(TIME_COL).reset_index(drop=True) if TIME_COL in df.columns else df.reset_index(drop=True)
    cut = int(len(d) * (1 - frac))
    return d.iloc[:cut].copy(), d.iloc[cut:].copy()


def split_clinic(subset: pd.DataFrame, train_frac: float, val_frac: float
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 70/15/15 within one clinic (same method as Step 1).

    Only the train part is loaded into the clinic DWH; val + test stay on the
    host for evaluation and (optionally) early stopping. Sorting by admission_ts
    makes it a clean past->future cut, mirroring a real deployment.
    """
    d = subset.sort_values(TIME_COL).reset_index(drop=True) if TIME_COL in subset.columns else subset.reset_index(drop=True)
    n = len(d)
    c1 = int(n * train_frac)
    c2 = int(n * (train_frac + val_frac))
    return d.iloc[:c1].copy(), d.iloc[c1:c2].copy(), d.iloc[c2:].copy()


@dataclass
class ClinicSpec:
    """Sampling specification for one simulated clinic."""
    site_id: int
    n_cases: int
    age_weights: dict[str, float] | None = None       # band -> weight
    gender_weights: dict[str, float] | None = None     # M/F -> weight
    triage_weights: dict[str, float] | None = None     # score -> weight


def _stratified_sample(
    df: pd.DataFrame,
    n: int,
    stratum_col: str,
    weights: dict[str, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample n rows with per-stratum target proportions.

    Strata not in `weights` are folded into an "OTHER" bucket.  If a stratum
    has fewer rows than its target allocation, all available rows are taken and
    the deficit is redistributed proportionally to the remaining strata.
    """
    strata = df[stratum_col].astype(str)
    known = set(weights)
    strata = strata.where(strata.isin(known), "OTHER")
    if "OTHER" not in weights:
        weights = dict(weights)
        weights["OTHER"] = 0.0

    total_w = sum(weights.values()) or 1.0
    targets = {k: max(1, int(round(n * v / total_w))) for k, v in weights.items() if v > 0}

    selected = []
    for stratum, target in targets.items():
        pool = df[strata == stratum]
        take = min(len(pool), target)
        if take > 0:
            selected.append(pool.sample(n=take, random_state=int(rng.integers(1 << 31))))

    result = pd.concat(selected, ignore_index=True) if selected else df.head(0)

    # top-up or trim to exactly n
    if len(result) < n:
        remaining = df.drop(result.index, errors="ignore")
        extra = min(n - len(result), len(remaining))
        if extra > 0:
            result = pd.concat([result, remaining.sample(n=extra, random_state=int(rng.integers(1 << 31)))])
    elif len(result) > n:
        result = result.sample(n=n, random_state=int(rng.integers(1 << 31)))

    return result.reset_index(drop=True)


def _age_band(age: float) -> str:
    if age < 18:
        return "0-17"
    if age < 40:
        return "18-39"
    if age < 65:
        return "40-64"
    return "65+"


def sample_clinic(
    df: pd.DataFrame,
    spec: ClinicSpec,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw a clinic subset according to the spec.

    Sampling is always *case-based* (whole encounters), never single-value
    overwrites — keeps clinical plausibility, as the assignment requires.
    """
    pool = df.copy()
    n = min(spec.n_cases, len(pool))

    # Level 2: age stratification
    if spec.age_weights:
        pool["_age_band"] = pool["age_in_years"].apply(_age_band)
        pool = _stratified_sample(pool, n, "_age_band", spec.age_weights, rng)
        pool = pool.drop(columns=["_age_band"], errors="ignore")
    # Level 3: gender stratification
    elif spec.gender_weights:
        pool = _stratified_sample(pool, n, "gender", spec.gender_weights, rng)
    # Level 4: triage stratification
    elif spec.triage_weights:
        pool["_triage_str"] = pool["triage_score"].astype(str)
        pool = _stratified_sample(pool, n, "_triage_str", spec.triage_weights, rng)
        pool = pool.drop(columns=["_triage_str"], errors="ignore")
    # Level 1 (or combined fallback): pure random
    else:
        pool = pool.sample(n=n, random_state=int(rng.integers(1 << 31)))

    return pool.reset_index(drop=True)


# ================================================================ scenario builders

def build_level1(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> list[ClinicSpec]:
    """Level 1: different case volumes only."""
    sizes = cfg.get("sizes", [2000, 1500, 1000, 500, 200])
    return [ClinicSpec(site_id=i, n_cases=s) for i, s in enumerate(sizes)]


def build_level2(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> list[ClinicSpec]:
    """Level 2: different age distributions, equal volume."""
    n = cfg.get("n_per_site", 1000)
    profiles = cfg.get("profiles", [
        {"0-17": 0.3, "18-39": 0.3, "40-64": 0.2, "65+": 0.2},  # young-heavy
        {"0-17": 0.1, "18-39": 0.2, "40-64": 0.3, "65+": 0.4},  # elderly-heavy
        {"0-17": 0.2, "18-39": 0.3, "40-64": 0.3, "65+": 0.2},  # balanced
        {"0-17": 0.05, "18-39": 0.15, "40-64": 0.3, "65+": 0.5}, # geriatric
        {"0-17": 0.25, "18-39": 0.25, "40-64": 0.25, "65+": 0.25}, # uniform
    ])
    return [ClinicSpec(site_id=i, n_cases=n, age_weights=p) for i, p in enumerate(profiles)]


def build_level3(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> list[ClinicSpec]:
    """Level 3: different gender distributions, equal volume."""
    n = cfg.get("n_per_site", 1000)
    profiles = cfg.get("profiles", [
        {"M": 0.7, "F": 0.3},
        {"M": 0.3, "F": 0.7},
        {"M": 0.5, "F": 0.5},
        {"M": 0.6, "F": 0.4},
        {"M": 0.4, "F": 0.6},
    ])
    return [ClinicSpec(site_id=i, n_cases=n, gender_weights=p) for i, p in enumerate(profiles)]


def build_level4(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> list[ClinicSpec]:
    """Level 4: different MTS triage mix, equal volume."""
    n = cfg.get("n_per_site", 1000)
    profiles = cfg.get("profiles", [
        {"1": 0.15, "2": 0.35, "3": 0.30, "4": 0.15, "5": 0.05},  # acute-heavy
        {"1": 0.02, "2": 0.10, "3": 0.30, "4": 0.40, "5": 0.18},  # low-acuity
        {"1": 0.05, "2": 0.20, "3": 0.40, "4": 0.25, "5": 0.10},  # standard
        {"1": 0.10, "2": 0.30, "3": 0.35, "4": 0.20, "5": 0.05},  # trauma center
        {"1": 0.03, "2": 0.15, "3": 0.35, "4": 0.30, "5": 0.17},  # suburban
    ])
    return [ClinicSpec(site_id=i, n_cases=n, triage_weights=p) for i, p in enumerate(profiles)]


def build_combined(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> list[ClinicSpec]:
    """Combined: different volume + age + gender + triage simultaneously."""
    specs_raw = cfg.get("sites", [])
    specs = []
    for i, s in enumerate(specs_raw):
        specs.append(ClinicSpec(
            site_id=i,
            n_cases=s.get("n_cases", 1000),
            age_weights=s.get("age_weights"),
            gender_weights=s.get("gender_weights"),
            triage_weights=s.get("triage_weights"),
        ))
    return specs


LEVEL_BUILDERS = {
    "level1": build_level1,
    "level2": build_level2,
    "level3": build_level3,
    "level4": build_level4,
    "combined": build_combined,
}


# ================================================================ run + manifest

def run_scenario(
    df: pd.DataFrame,
    scenario_name: str,
    scenario_cfg: dict,
    seed: int,
    out_dir: str,
    holdout_frac: float = 0.15,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict:
    """Generate all clinic subsets for one scenario, with proper splits.

    Layout written to out_dir/:
        global_holdout.parquet          rows in NO clinic (shared benchmark)
        site_<i>_train.parquet          -> loaded into clinic i's DWH
        site_<i>_val.parquet            -> host-side, early stopping (optional)
        site_<i>_test.parquet           -> host-side, per-clinic evaluation
        manifest.json                   reproducibility record
    """
    rng = np.random.default_rng(seed)
    level = scenario_cfg.get("level", scenario_name)
    builder = LEVEL_BUILDERS.get(level)
    if builder is None:
        raise ValueError(f"Unknown level '{level}', expected one of {list(LEVEL_BUILDERS)}")

    os.makedirs(out_dir, exist_ok=True)

    # 1. Carve the global held-out set BEFORE clinics are sampled (no leakage).
    pool, holdout = carve_global_holdout(df, holdout_frac, seed)
    holdout.to_parquet(os.path.join(out_dir, "global_holdout.parquet"), index=False)

    # 2. Sample clinics from the remaining pool only.
    specs = builder(pool, scenario_cfg, rng)

    manifest = {
        "scenario": scenario_name,
        "level": level,
        "seed": seed,
        "source_rows": len(df),
        "global_holdout_rows": len(holdout),
        "pool_rows": len(pool),
        "split": {"train_frac": train_frac, "val_frac": val_frac,
                  "test_frac": round(1 - train_frac - val_frac, 4),
                  "global_holdout_frac": holdout_frac},
        "n_sites": len(specs),
        "sites": [],
    }

    for spec in specs:
        site_rng = np.random.default_rng(seed + spec.site_id + 1)
        subset = sample_clinic(pool, spec, site_rng)

        # 3. Chronological 70/15/15 within the clinic.
        tr, va, te = split_clinic(subset, train_frac, val_frac)
        base = f"site_{spec.site_id}"
        tr.to_parquet(os.path.join(out_dir, f"{base}_train.parquet"), index=False)
        va.to_parquet(os.path.join(out_dir, f"{base}_val.parquet"), index=False)
        te.to_parquet(os.path.join(out_dir, f"{base}_test.parquet"), index=False)

        site_info = {
            "site_id": spec.site_id,
            "n_cases": len(subset),
            "n_train": len(tr),
            "n_val": len(va),
            "n_test": len(te),
            "admitted_rate": round(float(subset["admitted"].mean()), 4) if "admitted" in subset.columns else None,
            "train_file": f"{base}_train.parquet",
            "val_file": f"{base}_val.parquet",
            "test_file": f"{base}_test.parquet",
        }
        if "age_in_years" in subset.columns:
            site_info["mean_age"] = round(float(subset["age_in_years"].mean()), 1)
        if "gender" in subset.columns:
            site_info["male_frac"] = round(float((subset["gender"] == "M").mean()), 3)
        if "triage_score" in subset.columns:
            ts = pd.to_numeric(subset["triage_score"], errors="coerce")
            site_info["mean_triage"] = round(float(ts.mean()), 2) if ts.notna().any() else None
        manifest["sites"].append(site_info)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ================================================================ CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate simulated clinic datasets (Level 1-4)")
    ap.add_argument("--dump", required=True, help="path to the immutable parquet dump")
    ap.add_argument("--config", required=True, help="path to simulation config YAML")
    ap.add_argument("--out", default="data/simulated/", help="output root directory")
    ap.add_argument("--scenario", default=None,
                    help="run only this scenario (default: all in config)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_parquet(args.dump)
    if "admitted" not in df.columns and "verbleib" in df.columns:
        df["admitted"] = df["verbleib"].astype(str).str.startswith("Aufnahme").astype(int)
    print(f"Loaded dump: {len(df)} encounters")

    seed = cfg.get("seed", 42)
    scenarios = cfg.get("scenarios", {})
    run = {args.scenario: scenarios[args.scenario]} if args.scenario else scenarios

    for name, scfg in run.items():
        out_dir = os.path.join(args.out, name)
        manifest = run_scenario(df, name, scfg, seed, out_dir)
        print(f"\n{'='*50}")
        print(f"Scenario: {name} (level={manifest['level']}, seed={seed})")
        for s in manifest["sites"]:
            extra = []
            if "mean_age" in s:
                extra.append(f"age={s['mean_age']}")
            if "male_frac" in s:
                extra.append(f"male={s['male_frac']:.0%}")
            if "mean_triage" in s:
                extra.append(f"triage={s['mean_triage']}")
            if s.get("admitted_rate") is not None:
                extra.append(f"adm={s['admitted_rate']:.1%}")
            print(f"  site_{s['site_id']}: n={s['n_cases']:>5}  {', '.join(extra)}")
        print(f"  → {out_dir}/")


if __name__ == "__main__":
    main()
