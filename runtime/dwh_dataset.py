"""
dwh_dataset.py — turn a DWH dump (export_dwh_data.py output) into a model-ready dataset.

This is the "local data export -> training dataset" stage of the Step-1 pipeline. It is
deliberately explicit and transparent so the feature engineering can be copied straight
into the thesis Methods section.

Pipeline it implements:
  parquet dump (raw AKTIN columns)
    -> chronological split by admission_ts (70/15/15)
    -> Preprocessor.fit(train)   [medians, categorical vocab, scaler — TRAIN ONLY]
    -> Preprocessor.transform(train/val/test) -> (X, feature_names)

Design choices that matter for the thesis:
  * The Preprocessor is fit ONLY on the training split. Val/test/inference reuse it.
    This prevents any leakage from future data into the model inputs.
  * It is picklable and self-describing (feature_names, input_dim). The SAME object is
    reused unchanged in Step 2 for every federated site and for MELD inference, so the
    input space is identical everywhere.
  * Only continuous features are standardized; one-hot and binary columns pass through
    as 0/1, which keeps them interpretable.
  * Missingness of vitals/triage is kept as an explicit signal (a "<feat>_missing"
    column) before the value itself is median-imputed.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Columns produced by export_dwh_data.py (and make_synthetic_dump.py).
TIME_COL = "admission_ts"


@dataclass
class FeatureSpec:
    """Which raw columns become which kind of feature. Mirrors config/step1.yaml."""
    numeric: list[str] = field(default_factory=lambda: [
        "age_in_years", "respiratory_rate", "heart_rate", "oxygen_saturation",
        "body_temperature", "gcs_total_score", "pain_level",
        "waiting_time_minutes", "admission_hour",
    ])
    ordinal: list[str] = field(default_factory=lambda: ["triage_score"])
    categorical: list[str] = field(default_factory=lambda: [
        "gender", "referral_type", "time_of_day", "cedis_code",
    ])
    missingness_indicators: list[str] = field(default_factory=lambda: [
        "respiratory_rate", "heart_rate", "oxygen_saturation", "body_temperature",
        "gcs_total_score", "pain_level", "triage_score",
    ])
    cedis_min_freq: float = 0.01
    # isolation_status is ~100% missing in the data, so is_isolated is a dead
    # (always-0) feature. Dropped by default; set True to re-enable.
    include_isolation: bool = False
    target: str = "admitted"


# ---------------------------------------------------------------- load / target
def load_dump(path: str) -> pd.DataFrame:
    """Read the parquet dump and ensure the binary target column exists."""
    df = pd.read_parquet(path)
    if "admitted" not in df.columns:
        if "verbleib" not in df.columns:
            raise ValueError("dump has neither 'admitted' nor 'verbleib' to derive a target")
        df["admitted"] = df["verbleib"].astype(str).str.startswith("Aufnahme").astype(int)
    return df


def chronological_split(df: pd.DataFrame, train_frac: float = 0.70,
                        val_frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Earliest rows -> train, middle -> validation, latest -> test.

    Sorting by admission_ts makes the split a clean time-based cut, which matches how a
    deployed model would be trained on the past and applied to the future.
    """
    d = df.sort_values(TIME_COL).reset_index(drop=True) if TIME_COL in df.columns else df.reset_index(drop=True)
    n = len(d)
    c1 = int(n * train_frac)
    c2 = int(n * (train_frac + val_frac))
    return d.iloc[:c1].copy(), d.iloc[c1:c2].copy(), d.iloc[c2:].copy()


# ---------------------------------------------------------------- preprocessor
class Preprocessor:
    """Fit on TRAIN only; reuse for val/test/inference and (in Step 2) every site.

    Produces a fixed-order float32 matrix. Continuous block is standardized; the
    categorical/binary block is 0/1 passthrough. Feature order is frozen at fit time.
    """

    def __init__(self, spec: FeatureSpec | None = None):
        self.spec = spec or FeatureSpec()
        self.medians_: dict[str, float] = {}
        self.categories_: dict[str, list[str]] = {}
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self._num_cols: list[str] = []      # numeric + ordinal, order fixed at fit

    # -- helpers ------------------------------------------------------------
    def _to_num(self, df: pd.DataFrame, col: str) -> pd.Series:
        return pd.to_numeric(df.get(col), errors="coerce")

    def _cat_values(self, df: pd.DataFrame, col: str) -> pd.Series:
        s = df.get(col)
        if s is None:
            return pd.Series(["__MISSING__"] * len(df))
        return s.astype("object").where(s.notna(), "__MISSING__").astype(str)

    # -- fit ----------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        s = self.spec
        self._num_cols = s.numeric + s.ordinal

        # numeric medians (train only)
        for col in self._num_cols:
            vals = self._to_num(df, col)
            self.medians_[col] = float(vals.median()) if vals.notna().any() else 0.0

        # categorical vocabularies (train only); rare cedis folded to OTHER
        for col in s.categorical:
            vc = self._cat_values(df, col).value_counts(normalize=True)
            if col == "cedis_code":
                cats = sorted(vc[vc >= s.cedis_min_freq].index.tolist())
                if "OTHER" not in cats:
                    cats.append("OTHER")
            else:
                cats = sorted(vc.index.tolist())
            self.categories_[col] = cats

        # freeze feature order and fit the scaler on the numeric block
        self.feature_names_ = self._build_names()
        Xnum = self._numeric_block(df)
        self.scaler_ = StandardScaler().fit(Xnum)
        return self

    # -- transform ----------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("Preprocessor.transform called before fit")
        num = self.scaler_.transform(self._numeric_block(df)).astype(np.float32)
        cat = self._categorical_block(df).astype(np.float32)
        return np.concatenate([num, cat], axis=1)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # -- block builders -----------------------------------------------------
    def _numeric_block(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in self._num_cols:
            v = self._to_num(df, col).fillna(self.medians_[col])
            cols.append(v.to_numpy(dtype=np.float32))
        return np.column_stack(cols)

    def _categorical_block(self, df: pd.DataFrame) -> np.ndarray:
        s = self.spec
        blocks = []

        # missingness indicators (before imputation)
        for col in s.missingness_indicators:
            miss = self._to_num(df, col).isna().to_numpy(dtype=np.float32)
            blocks.append(miss.reshape(-1, 1))

        # engineered binaries
        wd = self._to_num(df, "admission_weekday")
        blocks.append((wd >= 6).fillna(False).to_numpy(dtype=np.float32).reshape(-1, 1))
        if s.include_isolation:
            iso = df.get("isolation_status")
            iso_flag = (iso.notna() if iso is not None else pd.Series([False] * len(df)))
            blocks.append(iso_flag.to_numpy(dtype=np.float32).reshape(-1, 1))

        # one-hot categoricals against the fixed train vocabulary
        for col in s.categorical:
            vals = self._cat_values(df, col)
            if col == "cedis_code":
                known = set(self.categories_[col])
                vals = vals.where(vals.isin(known), "OTHER")
            oh = np.zeros((len(df), len(self.categories_[col])), dtype=np.float32)
            index = {c: i for i, c in enumerate(self.categories_[col])}
            for r, v in enumerate(vals):
                j = index.get(v)
                if j is not None:
                    oh[r, j] = 1.0
            blocks.append(oh)

        return np.concatenate(blocks, axis=1)

    def _build_names(self) -> list[str]:
        s = self.spec
        names = list(self._num_cols)
        names += [f"{c}_missing" for c in s.missingness_indicators]
        names += ["is_weekend"]
        if s.include_isolation:
            names += ["is_isolated"]
        for col in s.categorical:
            names += [f"{col}={c}" for c in self.categories_[col]]
        return names

    @property
    def input_dim(self) -> int:
        return len(self.feature_names_)

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Preprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------- convenience
def build_xy(pre: Preprocessor, df: pd.DataFrame,
             target: str = "admitted") -> tuple[np.ndarray, np.ndarray]:
    """Transform features and pull the label array as float32."""
    X = pre.transform(df)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    return X, y


def describe_dataset(df: pd.DataFrame, target: str = "admitted") -> pd.DataFrame:
    """Small descriptive table for the Results section (Step 1)."""
    rows = [{
        "n_encounters": len(df),
        "admission_prevalence": round(float(pd.to_numeric(df[target]).mean()), 3),
        "mean_age": round(float(pd.to_numeric(df["age_in_years"], errors="coerce").mean()), 1),
        "median_waiting_min": round(float(pd.to_numeric(df["waiting_time_minutes"],
                                                        errors="coerce").median()), 1),
        "missing_triage": round(float(pd.to_numeric(df["triage_score"],
                                                    errors="coerce").isna().mean()), 3),
    }]
    return pd.DataFrame(rows)
