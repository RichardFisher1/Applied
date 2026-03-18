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
from pathlib import Path






class BatchTimeSeriesInspector:

    def __init__(self, df, time_col="Date and time", batch_col="Batch"):
        self.df = df.copy()
        self.time_col = time_col
        self.batch_col = batch_col

        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
        self.df = self.df.sort_values([self.batch_col, self.time_col])

        self.summary_df = None


    # -------------------------------------------------
    # Generate batch summary
    # -------------------------------------------------

    def summary(self, value_cols=None, only_missing=False):

        df = self.df

        if value_cols is None:
            value_cols = df.select_dtypes(include="number").columns.tolist()
            value_cols = [c for c in value_cols if c != self.batch_col]

        elif isinstance(value_cols, str):
            value_cols = [value_cols]

        rows = []

        for batch, g in df.groupby(self.batch_col):

            start = g[self.time_col].min()
            end = g[self.time_col].max()

            diffs = g[self.time_col].diff().dropna()
            resolution = diffs.mode().iloc[0] if not diffs.empty else pd.NaT

            if pd.notna(resolution):
                expected = pd.date_range(start=start, end=end, freq=resolution)
                missing = len(expected) - len(g)
            else:
                missing = None

            nan_counts = {
                f"nan_{col}": g[col].isna().sum()
                for col in value_cols if col in g.columns
            }

            rows.append({
                "Batch": batch,
                "start_time": start,
                "end_time": end,
                "duration": end - start,
                "resolution": resolution,
                "rows": len(g),
                "missing_timestamps": missing,
                **nan_counts
            })

        result = pd.DataFrame(rows)

        if only_missing:

            nan_cols = [c for c in result.columns if c.startswith("nan_")]

            mask_nan = result[nan_cols].sum(axis=1) > 0
            mask_missing = result["missing_timestamps"] > 0

            result = result[mask_nan | mask_missing]

        self.summary_df = result
        return result


    # -------------------------------------------------
    # Inspect one batch
    # -------------------------------------------------

    def inspect_batch(self, batch, include_nan_cols=True):

        if self.summary_df is None:
            raise ValueError("Run summary() first.")

        # allow single batch or list
        if isinstance(batch, (list, tuple, set)):
            rows = self.summary_df[self.summary_df["Batch"].isin(batch)].copy()
        else:
            rows = self.summary_df[self.summary_df["Batch"] == batch].copy()

        if rows.empty:
            print("Batch not found.")
            return None

        # --- identify nan columns ---
        nan_cols = [c for c in rows.columns if c.startswith("nan_")]

        # --- total missing ---
        rows["total_nan"] = rows[nan_cols].sum(axis=1)

        # --- optionally drop nan columns ---
        if not include_nan_cols:
            rows = rows.drop(columns=nan_cols)

        # --- remove all-zero columns ---
        rows = rows.loc[:, (rows != 0).any(axis=0)]

        return rows

    # -------------------------------------------------
    # Return raw batch data
    # -------------------------------------------------

    def get_batch(self, batch):

        return self.df[self.df[self.batch_col] == batch]


    # -------------------------------------------------
    # Sensors with NaNs in batch
    # -------------------------------------------------

    def nan_sensors(self, batch):

        row = self.inspect_batch(batch)

        if row is None:
            return None

        nan_cols = [c for c in row.columns if c.startswith("nan_")]

        return row[nan_cols].iloc[0][row[nan_cols].iloc[0] > 0]
    

    # -------------------------------------------------
    # Batches with clean data or issues
    # -------------------------------------------------

    def batches_by_quality(self, clean=True):

        if self.summary_df is None:
            raise ValueError("Run summary() first.")

        df = self.summary_df.copy()

        nan_cols = [c for c in df.columns if c.startswith("nan_")]

        # rows where all sensors have no NaNs
        no_nan = (df[nan_cols].sum(axis=1) == 0)

        # rows where timestamps are complete
        no_missing_ts = (df["missing_timestamps"] == 0)

        clean_mask = no_nan & no_missing_ts

        if clean:
            return df.loc[clean_mask, "Batch"].tolist()
        else:
            return df.loc[~clean_mask, "Batch"].tolist()
        


    # -------------------------------------------------
    # Dataset summary (simple overview)
    # -------------------------------------------------

    def dataset_summary(self):

        data = {
            "Metric": [
                "Rows",
                "Columns",
                "Number of batches",
                "Date start",
                "Date end",
                "Total NaNs",
            ],
            "Value": [
                len(self.df),
                len(self.df.columns),
                self.df[self.batch_col].nunique(),
                self.df[self.time_col].min(),
                self.df[self.time_col].max(),
                self.df.isna().sum().sum(),
            ],
        }

        return pd.DataFrame(data)


# ==========================================================
# Data Loading
# ==========================================================

def load_operating_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=[1, 2], low_memory=False)

    df["Date and time"] = pd.to_datetime(
        df["Date and time"],
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )

    df["Batch"] = pd.to_numeric(df["Batch"], errors="coerce").astype("Int64")

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

# ==========================================================
# Compute Mean Product
# ==========================================================
def compute_mean_product(prod_df: pd.DataFrame) -> pd.DataFrame:
    mean_prod = (
        prod_df.groupby("Batch")["Product"]
        .mean()
        .reset_index()
        .rename(columns={"Product": "mean_product"})
    )
    return mean_prod


# ==========================================================
# Batch Summary Features
# ==========================================================
def summarise_batches(df: pd.DataFrame) -> pd.DataFrame:
    """Compute batch-level summary statistics with totals for flow variables."""

    df = df.copy()

    flow_cols = [
        "LIQUID", "LIQUID.1", "LIQUID.2", "LIQUID.3", "LIQUID.4", "LIQUID.5",
        "GAS", "GAS.1", "GAS.2", "GAS.3"
    ]

    state_cols = [
        "pH",
        "OFFGAS", "OFFGAS.1",
        "PRESSURE", "PRESSURE.1",
        "OXYGEN"
    ]

    grouped = df.groupby("Batch")

    # Flow statistics (include totals)
    flow_stats = grouped[flow_cols].agg(["mean", "std", "max", "min", "sum"])

    # State statistics (no totals)
    state_stats = grouped[state_cols].agg(["mean", "std", "max", "min"])

    # Combine
    summary = pd.concat([flow_stats, state_stats], axis=1)

    # Flatten column names
    summary.columns = [f"{var}_{stat}" for var, stat in summary.columns]

    return summary.reset_index()


# ==========================================================
# Full Pipeline
# ==========================================================
def build_features_and_target(
    op_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    stats: Tuple[str, ...] = ("mean",),
    use_engineered_totals: bool = False,
    include_last_phase: bool = False,
    include_trends: bool = False,
    include_ranges: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:

    op_df = op_df.copy()

    # Optional engineered totals (normally OFF for independent sensors)
    if use_engineered_totals:
        ...

    # --------------------------------------------------
    # Base batch summary features
    # --------------------------------------------------
    summary_df = summarise_batches(op_df)
    feature_df = summary_df

    # --------------------------------------------------
    # Late phase features
    # --------------------------------------------------
    if include_last_phase:
       ...

    # --------------------------------------------------
    # Trend features
    # --------------------------------------------------
    if include_trends:
       ...

    # --------------------------------------------------
    # Range features
    # --------------------------------------------------
    if include_ranges:
        ...

    # --------------------------------------------------
    # Target variable (Mean Product)
    # --------------------------------------------------
    prod_mean_df = compute_mean_product(prod_df)

    # --------------------------------------------------
    # Merge features and target
    # --------------------------------------------------
    merged = feature_df.merge(prod_mean_df, on="Batch", how="left")

    features = merged.drop(columns=["mean_product"]).set_index("Batch")
    target = merged.set_index("Batch")["mean_product"]

    return features, target














# def add_productivity_rank_and_tier(
#     df: pd.DataFrame,
#     add_tier: bool = True,
#     high_cutoff: int = 7,
#     medium_cutoff: int = 14,
# ) -> pd.DataFrame:
#     """
#     Add productivity rank and optionally tier columns
#     using a fixed batch-to-rank mapping.
#     """

#     df = df.copy()

#     batch_to_productivity = {
#         4041: 1, 4043: 2, 4047: 3, 4040: 4, 4042: 5,
#         4046: 6, 4045: 7, 4052: 8, 4034: 9, 4044: 10,
#         4032: 11, 4030: 12, 4036: 13, 4033: 14, 4035: 15,
#         4048: 16, 4038: 17, 4039: 18, 4037: 19, 4051: 20,
#         4050: 21, 4053: 22
#     }

#     # Always add rank
#     df["productivity_rank"] = df["Batch"].map(batch_to_productivity)

#     # Add tier only if requested
#     if add_tier:

#         def rank_to_tier(rank):
#             if pd.isna(rank):
#                 return None
#             if rank <= high_cutoff:
#                 return "High"
#             elif rank <= medium_cutoff:
#                 return "Medium"
#             else:
#                 return "Low"

#         df["productivity_tier"] = df["productivity_rank"].apply(rank_to_tier)

#     return df

# # ==========================================================
# # Time Resampling Utilities
# # ==========================================================

# def resample_operating_timeseries(
#     df: pd.DataFrame,
#     rule: str = "1h",
#     how: str = "mean",
# ) -> pd.DataFrame:
#     """
#     Resample operating time-series to a coarser interval.
#     Works with MultiIndex containing 'Batch' and 'Date and time'.
#     """

#     if "Date and time" not in df.index.names:
#         raise ValueError("Index must contain 'Date and time' level")

#     if "Batch" not in df.index.names:
#         raise ValueError("Index must contain 'Batch' level")

#     # Group by all non-time levels
#     non_time_levels = [lvl for lvl in df.index.names if lvl != "Date and time"]

#     grouped = df.groupby(level=non_time_levels)

#     if how == "mean":
#         result = grouped.resample(rule, level="Date and time").mean()
#     elif how == "sum":
#         result = grouped.resample(rule, level="Date and time").sum()
#     elif how == "median":
#         result = grouped.resample(rule, level="Date and time").median()
#     else:
#         raise ValueError(f"Unsupported aggregation method: {how}")

#     return result.sort_index()


# # ==========================================================
# # Operating Data Structuring Pipeline
# # ==========================================================

# def prepare_operating_timeseries(
#     df: pd.DataFrame,
#     add_rank: bool = True,
#     add_tier: bool = True,
#     aggregate: bool = True,
#     resample_rule: str | None = None,
#     resample_how: str = "mean",
#     high_cutoff: int = 7,
#     medium_cutoff: int = 14,
# ) -> pd.DataFrame:

#     df = df.copy()

#     # --------------------------------------------------
#     # Optional ranking
#     # --------------------------------------------------
#     if add_rank:
#         df = add_productivity_rank_and_tier(
#             df,
#             add_tier=add_tier,
#             high_cutoff=high_cutoff,
#             medium_cutoff=medium_cutoff,
#         )

#     # --------------------------------------------------
#     # Optional signal aggregation (row-wise)
#     # --------------------------------------------------
#     if aggregate:
#         df = aggregate_operating_timeseries(df).reset_index()
#     else:
#         df = df.sort_values(["Batch", "Date and time"])

#     # --------------------------------------------------
#     # Build MultiIndex FIRST (required for resampling)
#     # --------------------------------------------------
#     index_cols = []

#     if add_rank:
#         index_cols.append("productivity_rank")
#         if add_tier:
#             index_cols.append("productivity_tier")

#     index_cols += ["Batch", "Date and time"]

#     df = df.set_index(index_cols).sort_index()

#     # --------------------------------------------------
#     # Optional resampling (NOW SAFE)
#     # --------------------------------------------------
#     if resample_rule is not None:
#         df = resample_operating_timeseries(
#             df,
#             rule=resample_rule,
#             how=resample_how,
#         )

#     return df


# def batch_time_diagnostics(df):

#     df = df.copy().reset_index()
#     df["Date and time"] = pd.to_datetime(df["Date and time"])

#     has_rank = "productivity_rank" in df.columns
#     has_tier = "productivity_tier" in df.columns

#     diagnostics = []

#     for batch_id, batch_df in df.groupby("Batch"):

#         batch_df = batch_df.sort_values("Date and time")

#         start = batch_df["Date and time"].min()
#         end = batch_df["Date and time"].max()

#         duration = end - start
#         duration_hours = duration.total_seconds() / 3600
#         duration_days = duration_hours / 24

#         calendar_days = (end.date() - start.date()).days + 1
#         actual_samples = batch_df["Date and time"].nunique()

#         time_diffs = batch_df["Date and time"].diff().dropna()

#         if len(time_diffs) == 0:
#             expected_samples = actual_samples
#         else:
#             dominant_delta = time_diffs.mode()[0]
#             expected_samples = int(duration / dominant_delta) + 1

#         missing_samples = expected_samples - actual_samples
#         percent_missing = (
#             100 * missing_samples / expected_samples
#             if expected_samples > 0 else 0
#         )

#         row = {
#             "Batch": batch_id,
#             "Start": start,
#             "End": end,
#             "Duration_hours": duration_hours,
#             "Duration_days": duration_days,
#             "Calendar_days_spanned": calendar_days,
#             "Expected_samples": expected_samples,
#             "Actual_samples": actual_samples,
#             "Missing_samples": missing_samples,
#             "Percent_missing": percent_missing,
#         }

#         if has_rank:
#             row["productivity_rank"] = batch_df["productivity_rank"].iloc[0]

#         if has_tier:
#             row["productivity_tier"] = batch_df["productivity_tier"].iloc[0]

#         diagnostics.append(row)

#     result = pd.DataFrame(diagnostics)

#     # -----------------------------
#     # Dynamic index construction
#     # -----------------------------
#     index_cols = []

#     if has_rank:
#         index_cols.append("productivity_rank")

#     if has_tier:
#         index_cols.append("productivity_tier")

#     index_cols.append("Batch")

#     result = result.set_index(index_cols)

#     # Optional sort
#     if has_rank:
#         result = result.sort_index()

#     return result


# # ==========================================================
# # Feature Engineering
# # ==========================================================

# def engineer_operating_totals(df: pd.DataFrame) -> pd.DataFrame:
#     """Create engineered total operating variables."""
#     df = df.copy()

#     liquid_cols = [c for c in df.columns if c.startswith("LIQUID")]
#     gas_cols = [c for c in df.columns if c.startswith("GAS")]
#     offgas_cols = [c for c in df.columns if c.startswith("OFFGAS")]
#     pressure_cols = [c for c in df.columns if c.startswith("PRESSURE")]

#     if liquid_cols:
#         df["TOTAL_LIQUID_INFLOW"] = df[liquid_cols].sum(axis=1)

#     if gas_cols:
#         df["TOTAL_GAS_INFLOW"] = df[gas_cols].sum(axis=1)

#     if offgas_cols:
#         df["TOTAL_OFFGAS"] = df[offgas_cols].mean(axis=1)

#     if pressure_cols:
#         df["MEAN_PRESSURE"] = df[pressure_cols].mean(axis=1)

#     # Drop original component columns
#     df = df.drop(columns=liquid_cols + gas_cols + offgas_cols + pressure_cols)

#     return df


# # ==========================================================
# # Target Computation
# # ==========================================================

# def compute_product_rate(
#     op_df: pd.DataFrame,
#     prod_df: pd.DataFrame
# ) -> pd.DataFrame:
#     """
#     Compute product rate per batch.

#     Product Rate (kg/hr) =
#         mean(Product [g/L]) *
#         mean(TOTAL_LIQUID_INFLOW [L/hr]) *
#         0.001
#     """

#     op_df = op_df.copy()

#     # Ensure total liquid inflow exists
#     if "TOTAL_LIQUID_INFLOW" not in op_df.columns:
#         liquid_cols = [c for c in op_df.columns if c.startswith("LIQUID")]
#         op_df["TOTAL_LIQUID_INFLOW"] = op_df[liquid_cols].sum(axis=1)

#     mean_product = (
#         prod_df.groupby("Batch")["Product"]
#         .mean()
#         .rename("mean_product")
#     )

#     mean_inflow = (
#         op_df.groupby("Batch")["TOTAL_LIQUID_INFLOW"]
#         .mean()
#         .rename("mean_inflow")
#     )

#     rate_df = pd.concat([mean_product, mean_inflow], axis=1).dropna()

#     rate_df["product_rate"] = (
#         rate_df["mean_product"] *
#         rate_df["mean_inflow"] *
#         0.001
#     )

#     return rate_df.reset_index()[["Batch", "product_rate"]]

# def summarise_last_phase(
#     df: pd.DataFrame,
#     fraction: float = 0.2
# ) -> pd.DataFrame:
#     """
#     Compute summary statistics over the last fraction of each batch.
#     """

#     results = []

#     for batch, group in df.groupby("Batch"):
#         group = group.sort_values("Date and time")
#         cutoff = int(len(group) * (1 - fraction))
#         late = group.iloc[cutoff:]

#         summary = late.mean(numeric_only=True)
#         summary["Batch"] = batch
#         results.append(summary)

#     late_df = pd.DataFrame(results)
#     return late_df


# import numpy as np

# def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Compute linear trend (slope) for each variable within each batch.
#     """

#     results = []

#     for batch, group in df.groupby("Batch"):
#         group = group.sort_values("Date and time")
#         slopes = {}

#         x = np.arange(len(group))

#         for col in group.columns:
#             if col not in ("Batch", "Date and time"):
#                 y = group[col].values
#                 if len(y) > 1 and not np.all(np.isnan(y)):
#                     slope = np.polyfit(x, y, 1)[0]
#                 else:
#                     slope = np.nan
#                 slopes[f"{col}_slope"] = slope

#         slopes["Batch"] = batch
#         results.append(slopes)

#     return pd.DataFrame(results)


# def compute_range_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Compute max-min range per batch for each variable.
#     """

#     measurement_cols = [c for c in df.columns if c not in ("Batch", "Date and time")]

#     grouped = df.groupby("Batch")[measurement_cols]

#     max_df = grouped.max()
#     min_df = grouped.min()

#     range_df = max_df - min_df
#     range_df.columns = [f"{c}_range" for c in range_df.columns]

#     return range_df.reset_index()


# def aggregate_operating_timeseries(
#     df: pd.DataFrame,
#     drop_components: bool = True
# ) -> pd.DataFrame:
#     """
#     Aggregate sensor streams into grouped signals
#     while preserving full time-series structure.
#     """

#     df = df.copy()

#     # Ensure sorted
#     df = df.sort_values(["Batch", "Date and time"])

#     # Identify column groups
#     liquid_cols = [c for c in df.columns if c.startswith("LIQUID")]
#     gas_cols = [c for c in df.columns if c.startswith("GAS")]
#     offgas_cols = [c for c in df.columns if c.startswith("OFFGAS")]
#     pressure_cols = [c for c in df.columns if c.startswith("PRESSURE")]

#     # Aggregate across columns (row-wise, no time aggregation)
#     if liquid_cols:
#         df["TOTAL_LIQUID_INFLOW"] = df[liquid_cols].sum(axis=1)

#     if gas_cols:
#         df["TOTAL_GAS_INFLOW"] = df[gas_cols].sum(axis=1)

#     if offgas_cols:
#         df["MEAN_OFFGAS"] = df[offgas_cols].mean(axis=1)

#     if pressure_cols:
#         df["MEAN_PRESSURE"] = df[pressure_cols].mean(axis=1)

#     if drop_components:
#         df = df.drop(
#             columns=liquid_cols + gas_cols + offgas_cols + pressure_cols,
#             errors="ignore"
#         )

#     # Return structured multi-index time-series
#     return df.set_index(["Batch", "Date and time"])


# # ==========================================================
# # Full Pipeline
# # ==========================================================
# def build_features_and_target(
#     op_df: pd.DataFrame,
#     prod_df: pd.DataFrame,
#     stats: Tuple[str, ...] = ("mean",),
#     use_engineered_totals: bool = False,
#     include_last_phase: bool = False,
#     include_trends: bool = False,
#     include_ranges: bool = False,
# ) -> Tuple[pd.DataFrame, pd.Series]:

#     op_df = op_df.copy()

#     # Optional engineered totals (normally OFF for independent sensors)
#     if use_engineered_totals:
#         op_df = engineer_operating_totals(op_df)

#     # --------------------------------------------------
#     # Base batch summary features
#     # --------------------------------------------------
#     summary_df = summarise_batches(op_df, stats=stats)
#     feature_df = summary_df

#     # --------------------------------------------------
#     # Late phase features
#     # --------------------------------------------------
#     if include_last_phase:
#         late_df = summarise_last_phase(op_df)
#         late_df = late_df.add_suffix("_late")
#         late_df = late_df.rename(columns={"Batch_late": "Batch"})
#         feature_df = feature_df.merge(late_df, on="Batch", how="left")

#     # --------------------------------------------------
#     # Trend features
#     # --------------------------------------------------
#     if include_trends:
#         trend_df = compute_trend_features(op_df)
#         feature_df = feature_df.merge(trend_df, on="Batch", how="left")

#     # --------------------------------------------------
#     # Range features
#     # --------------------------------------------------
#     if include_ranges:
#         range_df = compute_range_features(op_df)
#         feature_df = feature_df.merge(range_df, on="Batch", how="left")

#     # --------------------------------------------------
#     # Target variable
#     # --------------------------------------------------
#     rate_df = compute_product_rate(op_df, prod_df)

#     # --------------------------------------------------
#     # Merge features and target
#     # --------------------------------------------------
#     merged = feature_df.merge(rate_df, on="Batch", how="left")

#     features = merged.drop(columns=["product_rate"]).set_index("Batch")
#     target = merged.set_index("Batch")["product_rate"]

#     return features, target