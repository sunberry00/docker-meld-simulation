"""
export_dwh_data.py — Step 1, stage 1: DWH -> reproducible dump.

Extracts every protocol-defined predictor and the Verbleib target from ONE AKTIN i2b2
DWH and writes a single, immutable parquet "dump". That dump is the fixed starting point
for the rest of the pipeline (dwh_dataset.py -> train_local.py) and, later, for the
Level 1-4 clinic-simulation scripts. The dump must never be edited by hand — all
downstream changes go through scripts, so every dataset stays reproducible.

Temporal logic (no leakage): predictors are restricted to observations recorded up to
the first documented physician contact (+30 min tolerance); the target is the ED
disposition. Only information available at/before the first physician contact is used.

SQL below is unchanged from the validated concept mapping. Requires read access to the
i2b2crcdata schema. No DB? Use training/make_synthetic_dump.py to produce an identically
shaped dump and develop the whole pipeline offline.

    python export_dwh_data.py --db-uri postgresql://user:pw@host:5432/i2b2 \
                              --out data/raw/dwh_dump_seed_fixed.parquet
"""
import argparse

import pandas as pd
from sqlalchemy import create_engine, text

QUERY = """
        WITH
        -- 1. Basic patient data and process-related timestamp extractions
        EncounterBase AS (
            SELECT DISTINCT
                obs.encounter_num,
                obs.patient_num,
                EXTRACT(years FROM age(vis_dim.start_date, pat_dim.birth_date)) AS age_in_years,
                pat_dim.sex_cd AS gender,
                vis_dim.start_date AS admission_ts,
                -- Derived process variables from admission timestamp
                EXTRACT(HOUR FROM vis_dim.start_date) AS admission_hour,
                EXTRACT(ISODOW FROM vis_dim.start_date) AS admission_weekday, -- 1 = Monday, 7 = Sunday
                CASE 
                    WHEN EXTRACT(HOUR FROM vis_dim.start_date) >= 6 AND EXTRACT(HOUR FROM vis_dim.start_date) < 12 THEN 'Morning'
                    WHEN EXTRACT(HOUR FROM vis_dim.start_date) >= 12 AND EXTRACT(HOUR FROM vis_dim.start_date) < 18 THEN 'Afternoon'
                    WHEN EXTRACT(HOUR FROM vis_dim.start_date) >= 18 AND EXTRACT(HOUR FROM vis_dim.start_date) < 22 THEN 'Evening'
                    ELSE 'Night'
                END AS time_of_day
            FROM i2b2crcdata.observation_fact obs
            JOIN i2b2crcdata.patient_dimension pat_dim ON obs.patient_num = pat_dim.patient_num
            JOIN i2b2crcdata.visit_dimension vis_dim ON obs.encounter_num = vis_dim.encounter_num
        ),

        -- 2. TIME OF FIRST PHYSICIAN CONTACT (Strict temporal anchor for prediction)
        FirstContact AS (
            SELECT 
                encounter_num, 
                MIN(start_date) AS contact_ts
            FROM i2b2crcdata.observation_fact
            WHERE concept_cd = 'AKTIN:PHYSENCOUNTER' 
            GROUP BY encounter_num
        ),

        -- 3. ALL PREDICTORS (Captured up to the defined observation window)
        Predictors AS (
            SELECT
                o.encounter_num,
                
                -- Presentation and administrative context
                MAX(CASE
                    WHEN o.concept_cd = '75322-8:UNK' THEN '999'
                    WHEN o.concept_cd = 'CEDIS30:UNK' THEN '999'
                    WHEN o.concept_cd LIKE '75322-8%' THEN substr(o.concept_cd, 9)
                    WHEN o.concept_cd LIKE 'CEDIS30:%' THEN substr(o.concept_cd, 9)
                END) AS cedis_code,

                MAX(CASE
                    WHEN o.concept_cd LIKE 'MTS:%' THEN substr(o.concept_cd, 5)
                    WHEN o.concept_cd LIKE 'ESI:%' THEN substr(o.concept_cd, 5)
                END) AS triage_score,
                
                MAX(CASE 
                    WHEN o.concept_cd LIKE 'AKTIN:REFERRAL:%' THEN substr(o.concept_cd, 16) 
                END) AS referral_type,
                
                MAX(CASE 
                    WHEN o.concept_cd IN ('AKTIN:ISOLATION:ISO', 'AKTIN:ISOLATION:RISO', 'AKTIN:ISOLATION:ISO:NEG') 
                    THEN o.concept_cd 
                END) AS isolation_status,

                -- Complete set of vital signs as defined in the study protocol
                MAX(CASE WHEN o.concept_cd = 'LOINC:9279-1' THEN o.nval_num END) AS respiratory_rate,
                MAX(CASE WHEN o.concept_cd = 'LOINC:8867-4' THEN o.nval_num END) AS heart_rate,
                MAX(CASE WHEN o.concept_cd = 'LOINC:20564-1' THEN o.nval_num END) AS oxygen_saturation,
                MAX(CASE WHEN o.concept_cd = 'LOINC:8329-5' THEN o.nval_num END) AS body_temperature,
                MAX(CASE WHEN o.concept_cd = 'LOINC:9269-2' THEN o.nval_num END) AS gcs_total_score,
                MAX(CASE WHEN o.concept_cd = 'LOINC:72514-3' THEN o.nval_num END) AS pain_level

            FROM i2b2crcdata.observation_fact o
            JOIN FirstContact fc ON o.encounter_num = fc.encounter_num
            WHERE o.modifier_cd = '@'
              -- Ensure no post-contact clinical information is included point-forward
              AND o.start_date <= (fc.contact_ts + INTERVAL '30 minutes')
            GROUP BY o.encounter_num
        ),

        -- 4. TARGET VARIABLE (Emergency Department Disposition / Verbleib)
        Target AS (
            SELECT
                encounter_num,
                MAX(CASE
                    WHEN concept_cd = 'AKTIN:TRANSFER:1' THEN 'Aufnahme in Funktionsbereich'
                    WHEN concept_cd = 'AKTIN:TRANSFER:2' THEN 'Verlegung extern in Funktionsbereich'
                    WHEN concept_cd = 'AKTIN:TRANSFER:3' THEN 'Aufnahme auf Ueberwachungsstation'
                    WHEN concept_cd = 'AKTIN:TRANSFER:4' THEN 'Verlegung extern auf Ueberwachungsstation'
                    WHEN concept_cd = 'AKTIN:TRANSFER:5' THEN 'Aufnahme auf Normalstation'
                    WHEN concept_cd = 'AKTIN:TRANSFER:6' THEN 'Verlegung extern auf Normalstation'
                    WHEN concept_cd = 'AKTIN:DISCHARGE:1' THEN 'Tod'
                    WHEN concept_cd = 'AKTIN:DISCHARGE:2' THEN 'Entlassung gegen aerztlichen Rat'
                    WHEN concept_cd = 'AKTIN:DISCHARGE:3' THEN 'Behandlung durch Pat. abgebrochen'
                    WHEN concept_cd = 'AKTIN:DISCHARGE:4' THEN 'Entlassung nach Hause'
                    WHEN concept_cd = 'AKTIN:DISCHARGE:5' THEN 'Entlassung zu weiterbehandelnden Arzt'
                    WHEN concept_cd = 'AKTIN:DISCHARGE:6' THEN 'kein Arztkontakt'
                    WHEN concept_cd = 'AKTIN:DISCHARGE:OTH' THEN 'Sonstige Entlassung'
                END) AS verbleib
            FROM i2b2crcdata.observation_fact
            WHERE modifier_cd = '@'
              AND ((concept_cd LIKE '%TRANSFER%' AND concept_cd <> 'AKTIN:TRANSFER:ZeitpunktVerlegung') 
                   OR concept_cd LIKE '%DISCHARGE%')
            GROUP BY encounter_num
        )

        -- 5. FINAL DATA COMPILATION
        SELECT
            b.encounter_num,
            b.age_in_years,
            b.gender,
            b.admission_ts,
            fc.contact_ts,
            -- Process variables derived from the intervals
            EXTRACT(EPOCH FROM (fc.contact_ts - b.admission_ts))/60 AS waiting_time_minutes,
            b.admission_hour,
            b.admission_weekday,
            b.time_of_day,
            -- Extracted predictors
            p.cedis_code,
            p.triage_score,
            p.referral_type,
            p.isolation_status,
            p.respiratory_rate,
            p.heart_rate,
            p.oxygen_saturation,
            p.body_temperature,
            p.gcs_total_score,
            p.pain_level,
            -- Clinical target
            t.verbleib,
            -- Binary endpoint classification for the MLP engine
            (CASE WHEN t.verbleib LIKE 'Aufnahme%' THEN 1 ELSE 0 END) AS admitted
        FROM EncounterBase b
        JOIN FirstContact fc ON b.encounter_num = fc.encounter_num
        LEFT JOIN Predictors p ON b.encounter_num = p.encounter_num
        JOIN Target t ON b.encounter_num = t.encounter_num
        WHERE t.verbleib IS NOT NULL 
          AND t.verbleib <> 'kein Arztkontakt';
    """


def export_dwh_data(db_uri: str, output_path: str) -> None:
    engine = create_engine(db_uri)
    print("Exporting complete data pipeline from DWH...")
    with engine.connect() as conn:
        df = pd.read_sql(text(QUERY), conn)

    # Immutable dump — the fixed starting dataset for all downstream scripts.
    df.to_parquet(output_path, index=False)
    print(f"Full reproducible dump saved to {output_path} (Rows: {len(df)})")

    expected = ["respiratory_rate", "heart_rate", "oxygen_saturation",
                "body_temperature", "gcs_total_score", "pain_level",
                "waiting_time_minutes", "admission_hour", "admission_weekday", "time_of_day"]
    print("\nVerification of extracted feature counts:")
    print(df[expected].count())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export one AKTIN DWH to a reproducible dump")
    ap.add_argument("--db-uri", default="postgresql://i2b2crcdata:demouser@localhost:5432/i2b2",
                    help="SQLAlchemy DB URI for the i2b2 database")
    ap.add_argument("--out", default="data/raw/dwh_dump_seed_fixed.parquet",
                    help="output parquet path (the immutable dump)")
    args = ap.parse_args()
    export_dwh_data(args.db_uri, args.out)
