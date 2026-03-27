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

    def inspect_batch(self, batch, include_nan_cols=True):

        if self.summary_df is None:
            raise ValueError("Run summary() first.")

        if isinstance(batch, (list, tuple, set)):
            rows = self.summary_df[self.summary_df["Batch"].isin(batch)].copy()
        else:
            rows = self.summary_df[self.summary_df["Batch"] == batch].copy()

        if rows.empty:
            print("Batch not found.")
            return None

        nan_cols = [c for c in rows.columns if c.startswith("nan_")]

        rows["total_nan"] = rows[nan_cols].sum(axis=1)

        if not include_nan_cols:
            rows = rows.drop(columns=nan_cols)

        rows = rows.loc[:, (rows != 0).any(axis=0)]

        return rows

    def get_batch(self, batch):

        return self.df[self.df[self.batch_col] == batch]

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
    df = pd.read_excel(xlsx_path)

    df = df[df["Batch"].notna()].copy()

    if "Date and time" in df.columns:
        df["Date and time"] = pd.to_datetime(df["Date and time"], errors="coerce")

    df["Batch"] = pd.to_numeric(df["Batch"], errors="coerce").astype(int)
    df["Product"] = pd.to_numeric(df["Product"], errors="coerce")

    return df

def compute_mean_product(prod_df: pd.DataFrame) -> pd.DataFrame:
    mean_prod = (
        prod_df.groupby("Batch")["Product"]
        .mean()
        .reset_index()
        .rename(columns={"Product": "mean_product"})
    )
    return mean_prod

def summarise_batches(df: pd.DataFrame) -> pd.DataFrame:
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

    flow_stats = grouped[flow_cols].agg(["mean", "std", "max", "min", "sum"])

    state_stats = grouped[state_cols].agg(["mean", "std", "max", "min"])

    summary = pd.concat([flow_stats, state_stats], axis=1)

    summary.columns = [f"{var}_{stat}" for var, stat in summary.columns]

    return summary.reset_index()

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

