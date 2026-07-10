# Federated Learning Simulation on AKTIN DWH Data

Bachelor thesis project: prototypical evaluation of site-specific LoRA adapters
in a Docker-based federated learning simulation using emergency department
routine data. Built on a fork of [aktin/MELD](https://github.com/aktin/MELD),
extended with a training mode, an async orchestrator API, and an FL server.

## The experiment, end to end

```
                     ┌──────────────────────────────────────────────┐
 Phase 0 (once)      │  Source AKTIN DWH (~100k test records)       │
                     └──────────────────┬───────────────────────────┘
                                        │ pipeline/01_export_dump.py
                                        ▼
                     ┌──────────────────────────────────────────────┐
                     │  Immutable dump (data/dump.parquet)          │
                     └──────────────────┬───────────────────────────┘
                                        │ pipeline/02_make_clinic_dumps.py
 Phase 1                                ▼    (Level 1-4 shifts, seeded)
                     ┌──────────────────────────────────────────────┐
                     │  5 clinic dumps (site_0..site_4.parquet)     │
                     └──────────────────┬───────────────────────────┘
                                        │
 Phase 2   docker compose up:  5 x (Postgres-i2b2 + MELD orchestrator)
                                        │
 Phase 3                                ▼ pipeline/03_load_clinics.py
                     ┌──────────────────────────────────────────────┐
                     │  Each clinic DB filled with its dump         │
                     └──────────────────┬───────────────────────────┘
                                        │
 Phase 4-5           fl_server.py  ⇄  5 x MELD orchestrator API
                     N rounds: dispatch backbone → local training in
                     runtime containers → FedAvg. LoRA adapters never
                     leave their site.
                                        │
 Phase 6                                ▼
                     ┌──────────────────────────────────────────────┐
                     │  experiments/<scenario>/  (metrics, weights) │
                     └──────────────────────────────────────────────┘
                                        │
 Phase 7   docker compose down -v — all clinics destroyed.
           Only the results remain. Next scenario starts fresh.
```

One command runs Phases 1-7:

```bash
python run_experiment.py --config config/experiment.yaml
```

## Directory guide

| Path | What it is |
|---|---|
| `pipeline/01_export_dump.py` | Source DWH → immutable parquet dump (run once) |
| `pipeline/02_make_clinic_dumps.py` | Dump → 5 shifted clinic dumps (Level 1-4 heterogeneity) |
| `pipeline/03_load_clinics.py` | Clinic dump → clinic database (direct i2b2 SQL load) |
| `run_experiment.py` | Chains all phases of one experiment |
| `fl_server.py` | FL coordination loop: dispatch → poll → FedAvg → repeat |
| `config/simulation.yaml` | Shift definitions for all Level 1-4 scenarios |
| `config/experiment.yaml` | Parameters of one experiment run |
| `MELD/` | Forked MELD orchestrator + our extensions (see below) |
| `runtime/` | The training/inference runtime container (entry.py + model code) |
| `infra/pg-i2b2/init.sql` | Minimal i2b2 schema for the lightweight clinic DBs |
| `docker-compose.fl.yaml` | The 5-clinic simulation stack |
| `examples/` | Original MELD runtime examples (nn / sklearn / tfdf) |
| `docker-pg/` | Early shared-volume prototype (superseded by MELD/ModelEnvironment/train.py) |

## What was added to MELD (fork delta)

Upstream MELD is a single-site **inference** orchestrator. This fork adds
**training** and **federation** without touching the existing inference path:

| File | Change |
|---|---|
| `MELD/api.py` | NEW — async REST API: `POST /start` → job_id, `GET /status/{job_id}`, `GET /results/{job_id}/backbone` |
| `MELD/ModelEnvironment/train.py` | REWRITTEN — container-based training via shared volume (was a subprocess stub) |
| `MELD/main.py` | +`serve` command (starts the API via uvicorn) |
| `MELD/requirements.txt` | +fastapi, uvicorn, python-multipart |
| `runtime/entry.py` | NEW — one image, two modes: `MELD_MODE=inference` \| `train` |
| `fl_server.py` | NEW — FedAvg loop over the site APIs |

Design decisions worth knowing:

- **Backbone travels, adapters stay.** Each round the server sends the global
  backbone; sites return trained backbones + `n_samples`. LoRA adapter weights
  are stored in each site's `adapter_store` volume and are never transmitted.
- **Runtime containers stay stateless** (MELD convention). Adapter persistence
  is handled by the orchestrator, which injects `adapter.pt` into `/input`
  before each round and collects the updated one from `/output` after.
- **Lightweight clinics.** A full docker-aktin-dwh is Postgres + WildFly +
  Apache (~1-2 GB RAM each). Training only needs SQL access, so each simulated
  clinic is a bare `postgres:16-alpine` with the minimal i2b2 schema
  (`infra/pg-i2b2/init.sql`). Five clinics fit on a laptop.
- **Direct SQL load instead of CDA import.** `03_load_clinics.py` copies i2b2
  rows (observation_fact, visit_dimension, patient_dimension) from the source
  DWH filtered by each clinic's encounter list. The CDA interface is a data
  *ingestion* mechanism and is not what this study evaluates.

## Setup

```bash
# 1. Python dependencies (host side: pipeline + FL server)
pip install -r requirements.txt

# 2. Build the two images
docker build -t meld-orchestrator:fl ./MELD
docker build -t meld-runtime:fl      ./runtime

# 3. Create the immutable dump from your source DWH (Phase 0, once)
python pipeline/01_export_dump.py \
    --db-uri postgresql://i2b2crcdata:demouser@localhost:5432/i2b2 \
    --out data/dump.parquet
```

## Run one experiment

```bash
# Pick the scenario in config/experiment.yaml (level1_volume, level2_age,
# level3_gender, level4_triage, combined), then:
python run_experiment.py --config config/experiment.yaml
```

Everything is seeded; `experiments/<scenario>/` afterwards contains the clinic
manifest (who got which cases), per-round FL logs, and the final global
backbone. The clinics themselves are gone — `docker compose down -v` removes
containers and volumes.

To run phases manually instead (for debugging), every pipeline script has its
own CLI; see the docstring at the top of each file.

## Reproducibility

Every experiment records: the scenario, the seed, per-clinic case counts and
distributions (`clinic_dumps/manifest.json`), FL round logs with `n_samples`
per site, and the exact configs used. The source dump is created once and
never modified; all shifts are derived by seeded scripts.

## Upstream MELD usage (single-site inference)

The original MELD workflow (`docker compose run meld run|pull|delete`) is
unchanged — see `scripts/compose.yaml` and `docs/`.
