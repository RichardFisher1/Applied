# """
# data_processing.py
# ===================

# TEMPLATE - WILL UPDATE WITH ARE OWN WORK

# This module contains functions for loading and preprocessing the
# biotechnological process data provided for the case study. The raw
# datasets consist of two separate files:

# * A large ``4000 series operating data`` CSV file with high‑frequency
#   measurements from process sensors. Each row records values for a
#   particular timestamp and batch. The first two rows contain header
#   information (parameter names and units) and should be discarded.
# * A smaller ``4000 series product data`` Excel sheet with lab
#   measurements of product concentration collected roughly every four
#   hours. The first two rows in this file also contain header/unit
#   information.

# The goal of preprocessing is to compute a set of summary statistics
# per batch that can be used as features for downstream machine
# learning models. We also compute a target variable – the product
# rate – as instructed in the coursework specification:

# Author: Applied Bioinformatics case study team, 2026
# """

# from __future__ import annotations

# import os
# from typing import List, Tuple

# import numpy as np
# import pandas as pd


# def load_operating_data(csv_path: str) -> pd.DataFrame:
#     """Load and clean the operating data.

#     Parameters
#     ----------
#     csv_path : str
#         Path to the raw ``4000 series operating data`` CSV file.

#     Returns
#     -------
#     pd.DataFrame
#         A DataFrame containing only the rows with valid batch numbers
#         and numeric columns converted from strings where possible.

#     Notes
#     -----
#     The raw CSV contains two header rows: the first lists variable
#     names and the second lists units. These rows have missing values
#     in the ``Batch`` column. By dropping rows with missing ``Batch``
#     values we remove these unit rows. All remaining rows correspond
#     to process measurements.
#     """
#     df = pd.read_csv(csv_path, low_memory=False)
#     # Drop the first two rows which contain variable names and units.
#     df = df[df['Batch'].notna()].copy()
#     # Convert the timestamp column to pandas datetime if present.
#     if 'Date and time' in df.columns:
#         df['Date and time'] = pd.to_datetime(df['Date and time'], errors='coerce')
#     # Ensure batch is treated as integer where possible.
#     df['Batch'] = pd.to_numeric(df['Batch'], errors='coerce').astype(int)
#     # Identify numeric sensor columns (all except datetime and batch)
#     numeric_cols = [col for col in df.columns if col not in ('Date and time', 'Batch')]
#     # Coerce all measurement columns to numeric; non‑numeric values are set to NaN
#     for col in numeric_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     return df


# def load_product_data(xlsx_path: str) -> pd.DataFrame:
#     """Load and clean the product data.

#     Parameters
#     ----------
#     xlsx_path : str
#         Path to the raw ``4000 series product data`` Excel file.

#     Returns
#     -------
#     pd.DataFrame
#         A DataFrame containing only valid product measurements with
#         numeric product concentrations.
#     """
#     df = pd.read_excel(xlsx_path)
#     # Drop the first two rows (parameter names and units)
#     df = df[df['Batch'].notna()].copy()
#     if 'Date and time' in df.columns:
#         df['Date and time'] = pd.to_datetime(df['Date and time'], errors='coerce')
#     df['Batch'] = pd.to_numeric(df['Batch'], errors='coerce').astype(int)
#     # Coerce product concentration to numeric (g/litre)
#     df['Product'] = pd.to_numeric(df['Product'], errors='coerce')
#     return df


# def summarise_batches(df: pd.DataFrame, stats: Tuple[str, ...] = ("mean", "std")) -> pd.DataFrame:
#     """Summarise process variables for each batch.

#     Computes specified summary statistics (by default mean and standard
#     deviation) for every sensor column in the operating data. The
#     statistics are computed across time for each batch independently.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Cleaned operating data with numeric measurements.
#     stats : tuple of str
#         Names of pandas aggregation functions to apply. Typical
#         examples include ``"mean"``, ``"std"``, ``"median"``, ``"min"``, and
#         ``"max"``.

#     Returns
#     -------
#     pd.DataFrame
#         A DataFrame where each row corresponds to a batch and each
#         column contains a summary statistic for a particular
#         variable. Column names are formatted as
#         ``<variable>_<statistic>`` (e.g. ``pH_mean``).
#     """
#     measurement_cols = [col for col in df.columns if col not in ('Date and time', 'Batch')]
#     grouped = df.groupby('Batch')[measurement_cols]
#     # Compute the requested statistics
#     summary = grouped.agg(list(stats))
#     # Flatten the multi‑index on the columns to single level with
#     # <variable>_<statistic> naming convention
#     summary.columns = [f"{var}_{stat}" for var, stat in summary.columns]
#     return summary.reset_index()


# def engineer_operating_totals(df: pd.DataFrame) -> pd.DataFrame:
#     """Create engineered total/mean operating variables.

#     Creates:
#         - TOTAL_LIQUID_INFLOW
#         - TOTAL_GAS_INFLOW
#         - TOTAL_OFFGAS
#         - MEAN_PRESSURE

#     Then drops the original individual stream columns.
#     """

#     df = df.copy()

#     # --- Create totals ---

#     df["TOTAL_LIQUID_INFLOW"] = df[
#         ["LIQUID", "LIQUID.1", "LIQUID.2",
#          "LIQUID.3", "LIQUID.4", "LIQUID.5"]
#     ].sum(axis=1)

#     df["TOTAL_GAS_INFLOW"] = df[
#         ["GAS", "GAS.1", "GAS.2", "GAS.3"]
#     ].sum(axis=1)

#     df["TOTAL_OFFGAS"] = df[
#         ["OFFGAS", "OFFGAS.1"]
#     ].mean(axis=1)

#     df["MEAN_PRESSURE"] = df[
#         ["PRESSURE", "PRESSURE.1"]
#     ].mean(axis=1)

#     # --- Drop original columns ---

#     df = df.drop(columns=[
#         "LIQUID", "LIQUID.1", "LIQUID.2",
#         "LIQUID.3", "LIQUID.4", "LIQUID.5",
#         "GAS", "GAS.1", "GAS.2", "GAS.3",
#         "OFFGAS", "OFFGAS.1",
#         "PRESSURE", "PRESSURE.1"
#     ])

#     return df





# def compute_product_rate(op_df: pd.DataFrame, prod_df: pd.DataFrame) -> pd.DataFrame:
#     """Compute the product rate for each batch.

#     The product rate is defined by the coursework as the product of
#     (mean product concentration in g/L), (mean total liquid inflow in
#     L/hr), and a unit conversion factor of 0.001. The total liquid
#     inflow is calculated by summing the six individual liquid inflow
#     streams at each timestamp.

#     Parameters
#     ----------
#     op_df : pd.DataFrame
#         Cleaned operating data with numeric measurement columns and a
#         ``Batch`` column.
#     prod_df : pd.DataFrame
#         Cleaned product data with numeric product concentration and a
#         ``Batch`` column.

#     Returns
#     -------
#     pd.DataFrame
#         A DataFrame with one row per batch and a single column
#         ``product_rate`` containing the computed rate.
#     """
#     # Identify liquid inflow columns by name pattern
#     liquid_cols = [col for col in op_df.columns if col.startswith('LIQUID')]
#     # Precompute total inflow per row
#     total_inflow = op_df[liquid_cols].sum(axis=1)
#     op_df = op_df.copy()
#     op_df['total_inflow'] = total_inflow
#     rates = []
#     for batch in sorted(prod_df['Batch'].unique()):
#         # mean product (g/L)
#         mean_product = prod_df.loc[prod_df['Batch'] == batch, 'Product'].mean()
#         # mean total inflow (L/hr)
#         mean_total_inflow = op_df.loc[op_df['Batch'] == batch, 'total_inflow'].mean()
#         # compute rate
#         rate = mean_product * mean_total_inflow * 0.001
#         rates.append({'Batch': batch, 'product_rate': rate})
#     rate_df = pd.DataFrame(rates)
#     return rate_df


# def build_features_and_target(
#     operating_csv: str,
#     product_xlsx: str,
#     stats: Tuple[str, ...] = ("mean", "std"),
# ) -> Tuple[pd.DataFrame, pd.Series]:
#     """Load raw data, summarise features and compute the target.

#     Parameters
#     ----------
#     operating_csv : str
#         Path to the operating data CSV file.
#     product_xlsx : str
#         Path to the product data Excel file.
#     stats : tuple of str, optional
#         Summary statistics to compute for the features. The default
#         includes the mean and standard deviation. Additional statistics
#         may be added to enrich the feature set.

#     Returns
#     -------
#     features : pd.DataFrame
#         Feature matrix indexed by batch. Columns correspond to
#         summary statistics for each process variable.
#     target : pd.Series
#         Series of product rates (kg/hr) indexed by batch.
#     """
#     # Load and clean data
#     op_df = load_operating_data(operating_csv)
#     prod_df = load_product_data(product_xlsx)
#     # Build feature matrix
#     summary_df = summarise_batches(op_df, stats=stats)
#     # Compute target variable
#     rate_df = compute_product_rate(op_df, prod_df)
#     # Merge features and target on Batch
#     merged = summary_df.merge(rate_df, on='Batch', how='left')
#     # Set the index to batch for convenience
#     features = merged.drop(columns=['product_rate']).set_index('Batch')
#     target = merged.set_index('Batch')['product_rate']
#     return features, target

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


# ==========================================================
# Data Loading
# ==========================================================

def load_operating_data(csv_path: str) -> pd.DataFrame:
    """Load and clean operating data CSV."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Remove header/unit rows
    df = df[df["Batch"].notna()].copy()

    if "Date and time" in df.columns:
        df["Date and time"] = pd.to_datetime(df["Date and time"], errors="coerce")

    df["Batch"] = pd.to_numeric(df["Batch"], errors="coerce").astype(int)

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
# Feature Engineering
# ==========================================================

def engineer_operating_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered total operating variables."""
    df = df.copy()

    liquid_cols = [c for c in df.columns if c.startswith("LIQUID")]
    gas_cols = [c for c in df.columns if c.startswith("GAS")]
    offgas_cols = [c for c in df.columns if c.startswith("OFFGAS")]
    pressure_cols = [c for c in df.columns if c.startswith("PRESSURE")]

    if liquid_cols:
        df["TOTAL_LIQUID_INFLOW"] = df[liquid_cols].sum(axis=1)

    if gas_cols:
        df["TOTAL_GAS_INFLOW"] = df[gas_cols].sum(axis=1)

    if offgas_cols:
        df["TOTAL_OFFGAS"] = df[offgas_cols].mean(axis=1)

    if pressure_cols:
        df["MEAN_PRESSURE"] = df[pressure_cols].mean(axis=1)

    # Drop original component columns
    df = df.drop(columns=liquid_cols + gas_cols + offgas_cols + pressure_cols)

    return df


def summarise_batches(
    df: pd.DataFrame,
    stats: Tuple[str, ...] = ("mean",)
) -> pd.DataFrame:
    """Compute batch-level summary statistics."""
    measurement_cols = [c for c in df.columns if c not in ("Date and time", "Batch")]

    grouped = df.groupby("Batch")[measurement_cols]
    summary = grouped.agg(list(stats))

    summary.columns = [f"{var}_{stat}" for var, stat in summary.columns]

    return summary.reset_index()


# ==========================================================
# Target Computation
# ==========================================================

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


def summarise_last_phase(
    df: pd.DataFrame,
    fraction: float = 0.2
) -> pd.DataFrame:
    """
    Compute summary statistics over the last fraction of each batch.
    """

    results = []

    for batch, group in df.groupby("Batch"):
        group = group.sort_values("Date and time")
        cutoff = int(len(group) * (1 - fraction))
        late = group.iloc[cutoff:]

        summary = late.mean(numeric_only=True)
        summary["Batch"] = batch
        results.append(summary)

    late_df = pd.DataFrame(results)
    return late_df


import numpy as np

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






# ==========================================================
# Full Pipeline
# ==========================================================

def build_features_and_target(
    op_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    stats: Tuple[str, ...] = ("mean",),
    use_engineered_totals: bool = True,
    include_last_phase: bool = False,
    include_trends: bool = False,
    include_ranges: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
        if use_engineered_totals:
            op_df = engineer_operating_totals(op_df)

        # Base summary
        summary_df = summarise_batches(op_df, stats=stats)

        feature_df = summary_df

        # Late-phase features
        if include_last_phase:
            late_df = summarise_last_phase(op_df)
            late_df = late_df.add_suffix("_late")
            late_df = late_df.rename(columns={"Batch_late": "Batch"})
            feature_df = feature_df.merge(late_df, on="Batch", how="left")

        # Trend features
        if include_trends:
            trend_df = compute_trend_features(op_df)
            feature_df = feature_df.merge(trend_df, on="Batch", how="left")

        # Range features
        if include_ranges:
            range_df = compute_range_features(op_df)
            feature_df = feature_df.merge(range_df, on="Batch", how="left")

        # Compute target
        rate_df = compute_product_rate(op_df, prod_df)

        # Merge features and target
        merged = feature_df.merge(rate_df, on="Batch", how="left")

        features = merged.drop(columns=["product_rate"]).set_index("Batch")
        target = merged.set_index("Batch")["product_rate"]

        return features, target