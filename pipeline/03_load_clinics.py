"""
03_load_clinics.py — load a simulated clinic dump into a clinic DWH (i2b2 Postgres).

Instead of generating CDA documents and importing via the FHIR/SOAP interface,
this script copies i2b2 rows directly. The goal of the study is to test
federated training, not the CDA ingestion pipeline.

Flow:
    1. Read encounter_nums from the clinic's parquet (created by 02_make_clinic_dumps.py).
    2. Look up the matching patient_nums in the SOURCE DWH.
    3. Stream the matching rows of observation_fact / visit_dimension /
       patient_dimension from SOURCE into TARGET.

Two production realities this version handles explicitly:

  * SCHEMA DRIFT. A real AKTIN i2b2 has more columns than the minimal clinic
    schema (e.g. observation_fact.text_search_index). Only the INTERSECTION of
    source and target columns is copied; source-only columns are skipped and
    reported. This makes the loader robust against i2b2 version differences.

  * VOLUME. One encounter has ~100+ observation_fact rows, so a batch of 500
    encounters is tens of thousands of rows. Parameterized multi-row INSERTs
    explode (>1M bind parameters) — rows are therefore streamed in bounded
    chunks and written with PostgreSQL COPY, which has no parameter limits and
    is orders of magnitude faster.

Usage:
    python pipeline/03_load_clinics.py \\
        --site-parquet experiments/<scenario>/clinic_dumps/site_0.parquet \\
        --source-db postgresql://user:pw@localhost:5432/i2b2 \\
        --target-db postgresql://i2b2crcdata:demouser@localhost:5440/i2b2 \\
        --clear
"""
from __future__ import annotations

import argparse
import csv
import io

import pandas as pd
from sqlalchemy import create_engine, text

SCHEMA = "i2b2crcdata"

# Unquoted \N in CSV marks NULL for COPY. A genuine data value that is exactly
# the two characters backslash-N would be read back as NULL — that value does
# not occur in AKTIN i2b2 data.
NULL_SENTINEL = "\\N"

# information_schema data_type values that must be serialized as integers.
# pandas reads nullable integer columns as float64 ("7.0"), which COPY into an
# integer column rejects — so integral floats are cast back before writing.
INT_TYPES = {"smallint", "integer", "bigint"}

TABLES_BY_ENCOUNTER = ["observation_fact", "visit_dimension"]
TABLES_BY_PATIENT = ["patient_dimension"]


# --------------------------------------------------------------- id lookups

def _encounter_nums(parquet_path: str) -> list[int]:
    """Encounter ids selected for this clinic by the simulation script."""
    df = pd.read_parquet(parquet_path, columns=["encounter_num"])
    return sorted(df["encounter_num"].astype(int).unique().tolist())


def _patient_nums(source_engine, encounter_nums: list[int]) -> list[int]:
    """Look up the patient ids belonging to the selected encounters."""
    pnums: set[int] = set()
    with source_engine.connect() as conn:
        for i in range(0, len(encounter_nums), 1000):
            chunk = encounter_nums[i:i + 1000]
            placeholders = ",".join(str(int(e)) for e in chunk)
            rows = conn.execute(text(
                f"SELECT DISTINCT patient_num FROM {SCHEMA}.observation_fact "
                f"WHERE encounter_num IN ({placeholders})"
            ))
            pnums.update(r[0] for r in rows)
    return sorted(pnums)


# --------------------------------------------------------------- copy engine

def _table_columns(engine, table: str) -> dict[str, str]:
    """Ordered {column_name: data_type} of a table, from information_schema."""
    q = text(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = :s AND table_name = :t ORDER BY ordinal_position"
    )
    with engine.connect() as conn:
        rows = conn.execute(q, {"s": SCHEMA, "t": table}).fetchall()
    return {r[0]: r[1] for r in rows}


def _serialize(value, as_int: bool):
    """Map one cell to its COPY-csv representation (None/NaN/NaT -> \\N)."""
    if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        return NULL_SENTINEL
    if as_int and isinstance(value, float):
        return str(int(value))
    return value


def _copy_chunk(raw_conn, table: str, cols: list[str],
                int_flags: list[bool], df: pd.DataFrame) -> None:
    """Write one DataFrame chunk into the target table via COPY FROM STDIN."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in df.itertuples(index=False, name=None):
        writer.writerow([_serialize(v, f) for v, f in zip(row, int_flags)])
    buf.seek(0)
    col_list = ", ".join(cols)
    with raw_conn.cursor() as cur:
        cur.copy_expert(
            f"COPY {SCHEMA}.{table} ({col_list}) "
            f"FROM STDIN WITH (FORMAT csv, NULL '\\N')",
            buf,
        )


def _copy_table(source_engine, target_engine, table: str,
                filter_col: str, ids: list[int],
                id_batch: int = 1000, row_chunk: int = 50000) -> int:
    """Stream filtered rows of one table from source to target.

    ids are batched for the WHERE clause; each SELECT is additionally read in
    row chunks so memory stays bounded no matter how many observation rows an
    encounter batch expands to.
    """
    src_cols = _table_columns(source_engine, table)
    tgt_cols = _table_columns(target_engine, table)
    cols = [c for c in tgt_cols if c in src_cols]  # target order, intersection
    dropped = sorted(set(src_cols) - set(tgt_cols))
    if dropped:
        print(f"    (skipping source-only columns: {', '.join(dropped)})")

    int_flags = [tgt_cols[c] in INT_TYPES for c in cols]
    col_sql = ", ".join(cols)

    total = 0
    raw = target_engine.raw_connection()
    try:
        for i in range(0, len(ids), id_batch):
            chunk_ids = ids[i:i + id_batch]
            placeholders = ",".join(str(int(v)) for v in chunk_ids)
            query = (f"SELECT {col_sql} FROM {SCHEMA}.{table} "
                     f"WHERE {filter_col} IN ({placeholders})")
            for df in pd.read_sql(text(query), source_engine, chunksize=row_chunk):
                if df.empty:
                    continue
                _copy_chunk(raw, table, cols, int_flags, df)
                total += len(df)
        raw.commit()
    finally:
        raw.close()
    return total


# --------------------------------------------------------------- public API

def load_site(parquet_path: str, source_uri: str, target_uri: str,
              clear: bool = False) -> dict:
    """Load one simulated clinic into a target DWH. Returns per-table row counts."""
    source = create_engine(source_uri)
    target = create_engine(target_uri)

    enums = _encounter_nums(parquet_path)
    pnums = _patient_nums(source, enums)
    print(f"  Site: {len(enums)} encounters, {len(pnums)} patients")

    if clear:
        all_tables = ", ".join(
            f"{SCHEMA}.{t}" for t in TABLES_BY_ENCOUNTER + TABLES_BY_PATIENT
        )
        with target.connect() as conn:
            conn.execute(text(f"TRUNCATE {all_tables}"))
            conn.commit()
        print("  Cleared target tables")

    stats: dict[str, int] = {}
    for table in TABLES_BY_ENCOUNTER:
        n = _copy_table(source, target, table, "encounter_num", enums)
        stats[table] = n
        print(f"  {SCHEMA}.{table}: {n} rows")
    for table in TABLES_BY_PATIENT:
        n = _copy_table(source, target, table, "patient_num", pnums)
        stats[table] = n
        print(f"  {SCHEMA}.{table}: {n} rows")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Load a simulated clinic into an AKTIN DWH")
    ap.add_argument("--site-parquet", required=True)
    ap.add_argument("--source-db", required=True, help="source DWH URI (the big one)")
    ap.add_argument("--target-db", required=True, help="target DWH URI (the clinic)")
    ap.add_argument("--clear", action="store_true", help="truncate target tables first")
    args = ap.parse_args()
    load_site(args.site_parquet, args.source_db, args.target_db, args.clear)


if __name__ == "__main__":
    main()
