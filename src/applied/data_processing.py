"""
data_processing.py
===================

Preprocessing module for the biotechnological process case study.

This module:
- Loads and cleans operating and product data
- Engineers aggregated operating variables
- Computes batch-level summary statistics
- Computes product rate target variable
- Builds feature matrix and target vector for ML

Author: Applied Bioinformatics case study team, 2026
"""

from __future__ import annotations

from typing import Tuple
import pandas as pd
import numpy as np


def load_operating_data(csv_path: str) -> pd.DataFrame:
    """Load and clean operating data CSV."""
    df = pd.read_csv(csv_path, low_memory=False)

    df = df[df["Batch"].notna()].copy()

    if "Date and time" in df.columns:
        df["Date and time"] = pd.to_datetime(df["Date and time"], format="%d/%m/%Y %H:%M", errors="coerce")

    df["Batch"] = pd.to_numeric(df["Batch"], errors="coerce").astype(int)

    df = df[df["Date and time"].notna()].copy()

    numeric_cols = [c for c in df.columns if c not in ("Date and time", "Batch")]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def load_product_data(xlsx_path: str) -> pd.DataFrame:
    """Load and clean product data Excel."""
    df = pd.read_excel(xlsx_path)

    df = df[df["Batch"].notna()].copy()

    if "Date and time" in df.columns:
        df["Date and time"] = pd.to_datetime(df["Date and time"], errors="coerce")

    df["Batch"] = pd.to_numeric(df["Batch"], errors="coerce").astype(int)
    df["Product"] = pd.to_numeric(df["Product"], errors="coerce")

    return df

def compute_product_rate(
    op_df: pd.DataFrame,
    prod_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute product rate per batch.

    Product Rate (kg/hr) =
        mean(Product [g/L]) *
        mean(TOTAL_LIQUID_INFLOW [L/hr]) *
        0.001
    """

    op_df = op_df.copy()

    # Ensure total liquid inflow exists
    if "TOTAL_LIQUID_INFLOW" not in op_df.columns:
        liquid_cols = [c for c in op_df.columns if c.startswith("LIQUID")]
        op_df["TOTAL_LIQUID_INFLOW"] = op_df[liquid_cols].sum(axis=1)

    mean_product = (
        prod_df.groupby("Batch")["Product"]
        .mean()
        .rename("mean_product")
    )

    mean_inflow = (
        op_df.groupby("Batch")["TOTAL_LIQUID_INFLOW"]
        .mean()
        .rename("mean_inflow")
    )

    rate_df = pd.concat([mean_product, mean_inflow], axis=1).dropna()

    rate_df["product_rate"] = (
        rate_df["mean_product"] *
        rate_df["mean_inflow"] *
        0.001
    )

    return rate_df.reset_index()[["Batch", "product_rate"]]

def add_productivity_rank_and_tier(
    df: pd.DataFrame,
    add_tier: bool = True,
    high_cutoff: int = 7,
    medium_cutoff: int = 14,
) -> pd.DataFrame:
    """
    Add productivity rank and optionally tier columns
    using a fixed batch-to-rank mapping.
    """

    df = df.copy()

    batch_to_productivity = {
        4041: 1, 4043: 2, 4047: 3, 4040: 4, 4042: 5,
        4046: 6, 4045: 7, 4052: 8, 4034: 9, 4044: 10,
        4032: 11, 4030: 12, 4036: 13, 4033: 14, 4035: 15,
        4048: 16, 4038: 17, 4039: 18, 4037: 19, 4051: 20,
        4050: 21, 4053: 22
    }

    # Always add rank
    df["productivity_rank"] = df["Batch"].map(batch_to_productivity)

    # Add tier only if requested
    if add_tier:

        def rank_to_tier(rank):
            if pd.isna(rank):
                return None
            if rank <= high_cutoff:
                return "High"
            elif rank <= medium_cutoff:
                return "Medium"
            else:
                return "Low"

        df["productivity_tier"] = df["productivity_rank"].apply(rank_to_tier)

    return df

def resample_operating_timeseries(
    df: pd.DataFrame,
    rule: str = "1h",
    how: str = "mean",
) -> pd.DataFrame:
    """
    Resample operating time-series to a coarser interval.
    Works with MultiIndex containing 'Batch' and 'Date and time'.
    """

    if "Date and time" not in df.index.names:
        raise ValueError("Index must contain 'Date and time' level")

    if "Batch" not in df.index.names:
        raise ValueError("Index must contain 'Batch' level")

    # Group by all non-time levels
    non_time_levels = [lvl for lvl in df.index.names if lvl != "Date and time"]

    grouped = df.groupby(level=non_time_levels)

    if how == "mean":
        result = grouped.resample(rule, level="Date and time").mean()
    elif how == "sum":
        result = grouped.resample(rule, level="Date and time").sum()
    elif how == "median":
        result = grouped.resample(rule, level="Date and time").median()
    else:
        raise ValueError(f"Unsupported aggregation method: {how}")

    return result.sort_index()

def aggregate_operating_timeseries(
    df: pd.DataFrame,
    drop_components: bool = True
) -> pd.DataFrame:
    """
    Aggregate sensor streams into grouped signals
    while preserving full time-series structure.
    """

    df = df.copy()

    # Ensure sorted
    df = df.sort_values(["Batch", "Date and time"])

    # Identify column groups
    liquid_cols = [c for c in df.columns if c.startswith("LIQUID")]
    gas_cols = [c for c in df.columns if c.startswith("GAS")]
    offgas_cols = [c for c in df.columns if c.startswith("OFFGAS")]
    pressure_cols = [c for c in df.columns if c.startswith("PRESSURE")]

    # Aggregate across columns (row-wise, no time aggregation)
    if liquid_cols:
        df["TOTAL_LIQUID_INFLOW"] = df[liquid_cols].sum(axis=1)

    if gas_cols:
        df["TOTAL_GAS_INFLOW"] = df[gas_cols].sum(axis=1)

    if offgas_cols:
        df["MEAN_OFFGAS"] = df[offgas_cols].mean(axis=1)

    if pressure_cols:
        df["MEAN_PRESSURE"] = df[pressure_cols].mean(axis=1)

    if drop_components:
        df = df.drop(
            columns=liquid_cols + gas_cols + offgas_cols + pressure_cols,
            errors="ignore"
        )

    # Return structured multi-index time-series
    return df.set_index(["Batch", "Date and time"])

def prepare_operating_timeseries(
    df: pd.DataFrame,
    add_rank: bool = True,
    add_tier: bool = True,
    aggregate: bool = True,
    resample_rule: str | None = None,
    resample_how: str = "mean",
    high_cutoff: int = 7,
    medium_cutoff: int = 14,
) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------
    # Optional ranking
    # --------------------------------------------------
    if add_rank:
        df = add_productivity_rank_and_tier(
            df,
            add_tier=add_tier,
            high_cutoff=high_cutoff,
            medium_cutoff=medium_cutoff,
        )

    # --------------------------------------------------
    # Optional signal aggregation (row-wise)
    # --------------------------------------------------
    if aggregate:
        df = aggregate_operating_timeseries(df).reset_index()
    else:
        df = df.sort_values(["Batch", "Date and time"])

    # --------------------------------------------------
    # Build MultiIndex FIRST (required for resampling)
    # --------------------------------------------------
    index_cols = []

    if add_rank:
        index_cols.append("productivity_rank")
        if add_tier:
            index_cols.append("productivity_tier")

    index_cols += ["Batch", "Date and time"]

    df = df.set_index(index_cols).sort_index()

    # --------------------------------------------------
    # Optional resampling (NOW SAFE)
    # --------------------------------------------------
    if resample_rule is not None:
        df = resample_operating_timeseries(
            df,
            rule=resample_rule,
            how=resample_how,
        )

    return df

def batch_time_diagnostics(df):

    df = df.copy().reset_index()
    df["Date and time"] = pd.to_datetime(df["Date and time"])

    has_rank = "productivity_rank" in df.columns
    has_tier = "productivity_tier" in df.columns

    diagnostics = []

    for batch_id, batch_df in df.groupby("Batch"):

        batch_df = batch_df.sort_values("Date and time")

        start = batch_df["Date and time"].min()
        end = batch_df["Date and time"].max()

        duration = end - start
        duration_hours = duration.total_seconds() / 3600
        duration_days = duration_hours / 24

        calendar_days = (end.date() - start.date()).days + 1
        actual_samples = batch_df["Date and time"].nunique()

        time_diffs = batch_df["Date and time"].diff().dropna()

        if len(time_diffs) == 0:
            expected_samples = actual_samples
        else:
            dominant_delta = time_diffs.mode()[0]
            expected_samples = int(duration / dominant_delta) + 1

        missing_samples = expected_samples - actual_samples
        percent_missing = (
            100 * missing_samples / expected_samples
            if expected_samples > 0 else 0
        )

        row = {
            "Batch": batch_id,
            "Start": start,
            "End": end,
            "Duration_hours": duration_hours,
            "Duration_days": duration_days,
            "Calendar_days_spanned": calendar_days,
            "Expected_samples": expected_samples,
            "Actual_samples": actual_samples,
            "Missing_samples": missing_samples,
            "Percent_missing": percent_missing,
        }

        if has_rank:
            row["productivity_rank"] = batch_df["productivity_rank"].iloc[0]

        if has_tier:
            row["productivity_tier"] = batch_df["productivity_tier"].iloc[0]

        diagnostics.append(row)

    result = pd.DataFrame(diagnostics)

    # -----------------------------
    # Dynamic index construction
    # -----------------------------
    index_cols = []

    if has_rank:
        index_cols.append("productivity_rank")

    if has_tier:
        index_cols.append("productivity_tier")

    index_cols.append("Batch")

    result = result.set_index(index_cols)

    # Optional sort
    if has_rank:
        result = result.sort_index()

    return result

def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute linear trend (slope) for each variable within each batch.
    """

    results = []

    for batch, group in df.groupby("Batch"):
        group = group.sort_values("Date and time")
        slopes = {}

        x = np.arange(len(group))

        for col in group.columns:
            if col not in ("Batch", "Date and time"):
                y = group[col].values
                if len(y) > 1 and not np.all(np.isnan(y)):
                    slope = np.polyfit(x, y, 1)[0]
                else:
                    slope = np.nan
                slopes[f"{col}_slope"] = slope

        slopes["Batch"] = batch
        results.append(slopes)

    return pd.DataFrame(results)

def compute_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute max-min range per batch for each variable.
    """

    measurement_cols = [c for c in df.columns if c not in ("Batch", "Date and time")]

    grouped = df.groupby("Batch")[measurement_cols]

    max_df = grouped.max()
    min_df = grouped.min()

    range_df = max_df - min_df
    range_df.columns = [f"{c}_range" for c in range_df.columns]

    return range_df.reset_index()


import numpy as np
import pandas as pd


import numpy as np


# ======================================================
# Utility
# ======================================================

def clean_signal(batch_df, column):
    df = batch_df.sort_values("relative_step")

    s = df[column].astype(float)
    t = df["relative_step"].astype(float)

    mask = (~s.isna()) & (~t.isna())

    return s[mask].values, t[mask].values


def encode_generic_signal(batch_df, column, prefix):
    """
    Generic encoder for signals:
    - mean
    - slope
    - diff std
    """

    s, t = clean_signal(batch_df, column)

    # Require minimum length
    if len(s) < 3:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_slope": np.nan,
            f"{prefix}_diff_std": np.nan,
        }

    diff = np.diff(s)

    # Safe slope computation
    try:
        slope = np.polyfit(t, s, 1)[0]
    except Exception:
        slope = np.nan

    return {
        f"{prefix}_mean": np.mean(s),
        f"{prefix}_slope": slope,
        f"{prefix}_diff_std": np.std(diff),
    }


# ======================================================
# Signal Encoders
# ======================================================
import numpy as np


# ======================================================
# Utility
# ======================================================

def clean_signal(batch_df, column):
    df = batch_df.sort_values("relative_step")

    s = df[column].astype(float)
    t = df["relative_step"].astype(float)

    mask = (~s.isna()) & (~t.isna())

    return s[mask].values, t[mask].values


def encode_generic_signal(batch_df, column, prefix):

    s, t = clean_signal(batch_df, column)

    if len(s) < 5:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_slope": np.nan,
            f"{prefix}_early_mean": np.nan,
            f"{prefix}_late_mean": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_auc": np.nan,
            f"{prefix}_time_to_max": np.nan,
            f"{prefix}_high_fraction": np.nan,
        }

    n = len(s)
    early_cut = int(n * 0.2)
    late_cut = int(n * 0.8)

    early_s = s[:early_cut]
    late_s = s[late_cut:]

    try:
        slope = np.polyfit(t, s, 1)[0]
    except:
        slope = np.nan

    # normalized time to max (0–1)
    time_to_max = np.argmax(s) / n

    # high regime persistence (top quartile)
    high_threshold = np.percentile(s, 75)
    high_fraction = np.mean(s > high_threshold)

    return {
        f"{prefix}_mean": np.mean(s),
        f"{prefix}_std": np.std(s),
        f"{prefix}_slope": slope,
        f"{prefix}_early_mean": np.mean(early_s),
        f"{prefix}_late_mean": np.mean(late_s),
        f"{prefix}_range": np.max(s) - np.min(s),
        f"{prefix}_auc": np.trapezoid(s, t),
        f"{prefix}_time_to_max": time_to_max,
        f"{prefix}_high_fraction": high_fraction,
    }

# ======================================================
# Signal Wrappers
# ======================================================

def encode_total_liquid_inflow(batch_df):
    return encode_generic_signal(batch_df, "TOTAL_LIQUID_INFLOW", "liquid")


def encode_oxygen(batch_df):
    return encode_generic_signal(batch_df, "OXYGEN", "oxy")


def encode_total_gas_inflow(batch_df):
    return encode_generic_signal(batch_df, "TOTAL_GAS_INFLOW", "gas")


def encode_mean_offgas(batch_df):
    return encode_generic_signal(batch_df, "MEAN_OFFGAS", "offgas")


def encode_mean_pressure(batch_df):
    return encode_generic_signal(batch_df, "MEAN_PRESSURE", "pressure")


def encode_pH(batch_df):

    s, t = clean_signal(batch_df, "pH")

    if len(s) < 5:
        return {
            "ph_mean_abs_dev": np.nan,
            "ph_early_mean": np.nan,
            "ph_late_mean": np.nan,
            "ph_slope": np.nan,
            "ph_diff_std": np.nan,
            "ph_range": np.nan,
            "ph_time_out_of_band": np.nan,
        }

    n = len(s)
    early_cut = int(n * 0.2)
    late_cut = int(n * 0.8)

    early_s = s[:early_cut]
    late_s = s[late_cut:]

    deviation = s - np.mean(s)
    diff = np.diff(s)

    try:
        slope = np.polyfit(t, s, 1)[0]
    except:
        slope = np.nan

    # fraction outside tight band (control quality)
    tolerance = 0.1
    out_of_band = np.mean(np.abs(deviation) > tolerance)

    return {
        "ph_mean_abs_dev": np.mean(np.abs(deviation)),
        "ph_early_mean": np.mean(early_s),
        "ph_late_mean": np.mean(late_s),
        "ph_slope": slope,
        "ph_diff_std": np.std(diff),
        "ph_range": np.max(s) - np.min(s),
        "ph_time_out_of_band": out_of_band,
    }


# --------------------------------------------------
# Encoder Registry (Modular)
# --------------------------------------------------

ENCODERS = [
    encode_total_liquid_inflow,
    encode_pH,
    encode_oxygen,
    encode_total_gas_inflow,
    encode_mean_offgas,
    encode_mean_pressure,
]





# ==========================================================
# Full Pipeline
# ==========================================================

def build_features_and_target(
    op_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    resample_rule: str = "1h",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full modular feature engineering pipeline.

    Steps:
        1. Prepare operating timeseries
        2. Resample
        3. Add relative_step
        4. Encode signals per batch
        5. Compute product_rate
        6. Return X, y
    """

    # --------------------------------------------------
    # 1️⃣ Prepare & Resample Operating Data
    # --------------------------------------------------
    op_df = prepare_operating_timeseries(
        op_df,
        add_rank=True,
        add_tier=False,
        aggregate=True,
        resample_rule=resample_rule,
    )

    # flatten index for processing
    op_df = op_df.reset_index()

    # --------------------------------------------------
    # 2️⃣ Add Relative Time Index
    # --------------------------------------------------
    op_df = op_df.sort_values(["Batch", "Date and time"])

    op_df["relative_step"] = (
        op_df.groupby("Batch")
        .cumcount()
    )

    # --------------------------------------------------
    # 3️⃣ Encode Per Batch
    # --------------------------------------------------
    feature_rows = []

    for batch_id, batch_df in op_df.groupby("Batch"):

        row = {}

        for encoder in ENCODERS:
            row.update(encoder(batch_df))

        row["Batch"] = batch_id

        feature_rows.append(row)

    feature_df = pd.DataFrame(feature_rows)

    # --------------------------------------------------
    # 4️⃣ Compute Target
    # --------------------------------------------------
    rate_df = compute_product_rate(
        op_df[["Batch", "TOTAL_LIQUID_INFLOW"]],
        prod_df
    )

    # --------------------------------------------------
    # 5️⃣ Merge Features & Target
    # --------------------------------------------------
    merged = feature_df.merge(rate_df, on="Batch", how="inner")

    X = merged.drop(columns=["product_rate"]).set_index("Batch")
    y = merged.set_index("Batch")["product_rate"]

    return X, y


