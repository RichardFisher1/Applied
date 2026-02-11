"""
data_processing.py
===================

TEMPLATE - WILL UPDATE WITH ARE OWN WORK

This module contains functions for loading and preprocessing the
biotechnological process data provided for the case study. The raw
datasets consist of two separate files:

* A large ``4000 series operating data`` CSV file with high‑frequency
  measurements from process sensors. Each row records values for a
  particular timestamp and batch. The first two rows contain header
  information (parameter names and units) and should be discarded.
* A smaller ``4000 series product data`` Excel sheet with lab
  measurements of product concentration collected roughly every four
  hours. The first two rows in this file also contain header/unit
  information.

The goal of preprocessing is to compute a set of summary statistics
per batch that can be used as features for downstream machine
learning models. We also compute a target variable – the product
rate – as instructed in the coursework specification:

Author: Applied Bioinformatics case study team, 2026
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_operating_data(csv_path: str) -> pd.DataFrame:
    """Load and clean the operating data.

    Parameters
    ----------
    csv_path : str
        Path to the raw ``4000 series operating data`` CSV file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the rows with valid batch numbers
        and numeric columns converted from strings where possible.

    Notes
    -----
    The raw CSV contains two header rows: the first lists variable
    names and the second lists units. These rows have missing values
    in the ``Batch`` column. By dropping rows with missing ``Batch``
    values we remove these unit rows. All remaining rows correspond
    to process measurements.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    # Drop the first two rows which contain variable names and units.
    df = df[df['Batch'].notna()].copy()
    # Convert the timestamp column to pandas datetime if present.
    if 'Date and time' in df.columns:
        df['Date and time'] = pd.to_datetime(df['Date and time'], errors='coerce')
    # Ensure batch is treated as integer where possible.
    df['Batch'] = pd.to_numeric(df['Batch'], errors='coerce').astype(int)
    # Identify numeric sensor columns (all except datetime and batch)
    numeric_cols = [col for col in df.columns if col not in ('Date and time', 'Batch')]
    # Coerce all measurement columns to numeric; non‑numeric values are set to NaN
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_product_data(xlsx_path: str) -> pd.DataFrame:
    """Load and clean the product data.

    Parameters
    ----------
    xlsx_path : str
        Path to the raw ``4000 series product data`` Excel file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only valid product measurements with
        numeric product concentrations.
    """
    df = pd.read_excel(xlsx_path)
    # Drop the first two rows (parameter names and units)
    df = df[df['Batch'].notna()].copy()
    if 'Date and time' in df.columns:
        df['Date and time'] = pd.to_datetime(df['Date and time'], errors='coerce')
    df['Batch'] = pd.to_numeric(df['Batch'], errors='coerce').astype(int)
    # Coerce product concentration to numeric (g/litre)
    df['Product'] = pd.to_numeric(df['Product'], errors='coerce')
    return df


def summarise_batches(df: pd.DataFrame, stats: Tuple[str, ...] = ("mean", "std")) -> pd.DataFrame:
    """Summarise process variables for each batch.

    Computes specified summary statistics (by default mean and standard
    deviation) for every sensor column in the operating data. The
    statistics are computed across time for each batch independently.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned operating data with numeric measurements.
    stats : tuple of str
        Names of pandas aggregation functions to apply. Typical
        examples include ``"mean"``, ``"std"``, ``"median"``, ``"min"``, and
        ``"max"``.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to a batch and each
        column contains a summary statistic for a particular
        variable. Column names are formatted as
        ``<variable>_<statistic>`` (e.g. ``pH_mean``).
    """
    measurement_cols = [col for col in df.columns if col not in ('Date and time', 'Batch')]
    grouped = df.groupby('Batch')[measurement_cols]
    # Compute the requested statistics
    summary = grouped.agg(list(stats))
    # Flatten the multi‑index on the columns to single level with
    # <variable>_<statistic> naming convention
    summary.columns = [f"{var}_{stat}" for var, stat in summary.columns]
    return summary.reset_index()


def compute_product_rate(op_df: pd.DataFrame, prod_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the product rate for each batch.

    The product rate is defined by the coursework as the product of
    (mean product concentration in g/L), (mean total liquid inflow in
    L/hr), and a unit conversion factor of 0.001. The total liquid
    inflow is calculated by summing the six individual liquid inflow
    streams at each timestamp.

    Parameters
    ----------
    op_df : pd.DataFrame
        Cleaned operating data with numeric measurement columns and a
        ``Batch`` column.
    prod_df : pd.DataFrame
        Cleaned product data with numeric product concentration and a
        ``Batch`` column.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per batch and a single column
        ``product_rate`` containing the computed rate.
    """
    # Identify liquid inflow columns by name pattern
    liquid_cols = [col for col in op_df.columns if col.startswith('LIQUID')]
    # Precompute total inflow per row
    total_inflow = op_df[liquid_cols].sum(axis=1)
    op_df = op_df.copy()
    op_df['total_inflow'] = total_inflow
    rates = []
    for batch in sorted(prod_df['Batch'].unique()):
        # mean product (g/L)
        mean_product = prod_df.loc[prod_df['Batch'] == batch, 'Product'].mean()
        # mean total inflow (L/hr)
        mean_total_inflow = op_df.loc[op_df['Batch'] == batch, 'total_inflow'].mean()
        # compute rate
        rate = mean_product * mean_total_inflow * 0.001
        rates.append({'Batch': batch, 'product_rate': rate})
    rate_df = pd.DataFrame(rates)
    return rate_df


def build_features_and_target(
    operating_csv: str,
    product_xlsx: str,
    stats: Tuple[str, ...] = ("mean", "std"),
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load raw data, summarise features and compute the target.

    Parameters
    ----------
    operating_csv : str
        Path to the operating data CSV file.
    product_xlsx : str
        Path to the product data Excel file.
    stats : tuple of str, optional
        Summary statistics to compute for the features. The default
        includes the mean and standard deviation. Additional statistics
        may be added to enrich the feature set.

    Returns
    -------
    features : pd.DataFrame
        Feature matrix indexed by batch. Columns correspond to
        summary statistics for each process variable.
    target : pd.Series
        Series of product rates (kg/hr) indexed by batch.
    """
    # Load and clean data
    op_df = load_operating_data(operating_csv)
    prod_df = load_product_data(product_xlsx)
    # Build feature matrix
    summary_df = summarise_batches(op_df, stats=stats)
    # Compute target variable
    rate_df = compute_product_rate(op_df, prod_df)
    # Merge features and target on Batch
    merged = summary_df.merge(rate_df, on='Batch', how='left')
    # Set the index to batch for convenience
    features = merged.drop(columns=['product_rate']).set_index('Batch')
    target = merged.set_index('Batch')['product_rate']
    return features, target