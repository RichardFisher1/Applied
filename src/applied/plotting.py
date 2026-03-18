import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from mpl_toolkits.mplot3d import Axes3D



class BatchTimeSeriesPlotter1:

    def __init__(self, df, time_col="Date and time", batch_col="Batch"):
        
        self.df = df.copy()
        self.time_col = time_col
        self.batch_col = batch_col

        self.df[time_col] = pd.to_datetime(self.df[time_col])

    def _prepare_batch(self, batch, freq):

        df_batch = self.df[self.df[self.batch_col] == batch].copy()

        if df_batch.empty:
            return None, None, None

        df_batch = df_batch.sort_values(self.time_col)

        duplicate_times = df_batch.loc[
            df_batch[self.time_col].duplicated(keep=False),
            self.time_col
        ]

        df_batch = df_batch.groupby(self.time_col).mean(numeric_only=True)

        if df_batch.index.empty:
            return None, None, None

        start = df_batch.index.min()
        end = df_batch.index.max()

        if pd.isna(start) or pd.isna(end):
            return None, None, None

        full_index = pd.date_range(
            start=start,
            end=end,
            freq=freq
        )

        missing_timestamps = full_index.difference(df_batch.index)

        df_batch = df_batch.reindex(full_index)

        return df_batch, duplicate_times, missing_timestamps

    def plot(self, batch=None, column=None, columns=None, batches=None, freq="15min", width=None):

        if batch is not None and column is not None:
            columns = column

        if isinstance(columns, str):
            columns = [columns]

        if isinstance(column, list):
            columns = column
            column = None

        # MODE 1: batch → multiple columns
        if batch is not None:

            if isinstance(batch, (int,float)):
                batches = [batch]
            else:
                batches = batch

            if columns is None:
                raise ValueError("columns must be provided when batch is used")

            if isinstance(columns, str):
                columns = [columns]

            n = len(columns)

            if width is None:
                width = n

            ncols = min(width, n)
            nrows = math.ceil(n / ncols)

            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols,4*nrows), sharex=True)
            axes = np.atleast_1d(axes).flatten()

            for i, col in enumerate(columns):

                ax = axes[i]

                for b in batches:

                    result = self._prepare_batch(b, freq)

                    if result is None:
                        continue

                    df_batch, duplicate_times, missing_timestamps = result

                    if col not in df_batch.columns:
                        continue

                    series = df_batch[col].dropna()
                    series = df_batch[col].dropna()

                    if series.empty:
                        continue

                    # normalized time (0 → 1 across batch duration)
                    x = np.linspace(0, 1, len(series))

                    ax.plot(
                    x,
                        series.values,
                        linewidth=1.5,
                        alpha=0.85,
                        label=f"Batch {b}"
                    )

                ax.set_title(col)
                ax.legend()

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle("Batch Comparison")
            plt.tight_layout()
            plt.show()

        # MODE 2: column → multiple batches
        elif column is not None:

            if batches is None:
                batches = sorted(self.df[self.batch_col].dropna().unique())

            if isinstance(batches, (int,float)):
                batches = [batches]

            n = len(batches)

            if width is None:
                width = n

            ncols = min(width, n)
            nrows = math.ceil(n / ncols)

            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols,4*nrows))
            axes = np.atleast_1d(axes).flatten()

            for i, batch in enumerate(batches):

                ax = axes[i]

                df_batch, duplicate_times, missing_timestamps = self._prepare_batch(batch, freq)

                if df_batch is None:
                    continue

                raw_series = df_batch[column]
                interp_series = raw_series.interpolate()

                ax.plot(df_batch.index, raw_series, color="black")

                nan_times = df_batch.index[df_batch[column].isna()]

                ax.scatter(
                    nan_times,
                    interp_series.loc[nan_times],
                    color="orange", s=20
                )

                ax.scatter(
                    missing_timestamps,
                    interp_series.reindex(missing_timestamps),
                    color="red", s=25
                )

                ax.scatter(
                    duplicate_times,
                    interp_series.loc[duplicate_times],
                    color="green", s=25
                )

                ax.set_title(f"Batch {batch}")

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle(column)
            plt.tight_layout()
            plt.show()

        else:
            raise ValueError("Provide either batch=... or column=...")

    def plot_batches_grid(self, columns, batches=None, freq="15min"):

        if isinstance(columns, str):
            columns = [columns]

        if batches is None:
            batches = sorted(self.df[self.batch_col].dropna().unique())

        if isinstance(batches, (int, float)):
            batches = [batches]

        nrows = len(columns)
        ncols = len(batches)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5*ncols, 3*nrows),
            sharex=False
        )

        axes = np.array(axes).reshape(nrows, ncols)

        for col_i, col in enumerate(columns):

            for batch_i, batch in enumerate(batches):

                ax = axes[col_i, batch_i]

                df_batch, duplicate_times, missing_timestamps = self._prepare_batch(batch, freq)

                if df_batch is None:
                    continue

                raw_series = df_batch[col]
                interp_series = raw_series.interpolate()

                ax.plot(df_batch.index, raw_series, color="black", linewidth=1)

                nan_times = df_batch.index[df_batch[col].isna()]

                ax.scatter(
                    nan_times,
                    interp_series.loc[nan_times],
                    color="orange",
                    s=15
                )

                ax.scatter(
                    missing_timestamps,
                    interp_series.reindex(missing_timestamps),
                    color="red",
                    s=20
                )

                ax.scatter(
                    duplicate_times,
                    interp_series.loc[duplicate_times],
                    color="green",
                    s=20
                )

                if col_i == 0:
                    ax.set_title(f"Batch {batch}")

                if batch_i == 0:
                    ax.set_ylabel(col)

        plt.tight_layout()
        plt.show()

    def plot_overlay(self, column, batches=None, freq="15min", normalize_time=True):

        if batches is None:
            batches = sorted(self.df[self.batch_col].dropna().unique())

        if isinstance(batches, (int, float)):
            batches = [batches]

        plt.figure(figsize=(10,5))

        for batch in batches:

            result = self._prepare_batch(batch, freq)

            if result is None:
                continue

            df_batch, duplicate_times, missing_timestamps = result

            if df_batch is None or df_batch.empty:
                continue

            if column not in df_batch.columns:
                continue

            series = df_batch[column].dropna()

            if series.empty:
                continue

            # normalized time 0 → 1
            if normalize_time:
                x = np.linspace(0, 1, len(series))
            else:
                x = series.index

            plt.plot(
                x,
                series.values,
                linewidth=1.5,
                alpha=0.85,
                label=f"Batch {batch}"
            )

        if normalize_time:
            plt.xlabel("Normalized Batch Progress (0 → 1)")
        else:
            plt.xlabel("Time")

        plt.ylabel(column)
        plt.title(column)

        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_batch_3d(self, batch, columns, freq="15min", normalize_time=True):

        if isinstance(columns, str):
            columns = [columns]

        df_batch, duplicate_times, missing_timestamps = self._prepare_batch(batch, freq)

        if df_batch is None or df_batch.empty:
            print("No data for batch")
            return

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')

        for i, col in enumerate(columns):

            if col not in df_batch.columns:
                continue

            series = df_batch[col].dropna()

            if series.empty:
                continue

            # X axis
            if normalize_time:
                x = np.linspace(0,1,len(series))
            else:
                x = series.index

            # Y axis = variable index
            y = np.full(len(series), i)

            # Z axis = value
            z = series.values

            ax.plot(x, y, z, linewidth=2)

        ax.set_yticks(range(len(columns)))
        ax.set_yticklabels(columns)

        if normalize_time:
            ax.set_xlabel("Normalized Batch Progress")
        else:
            ax.set_xlabel("Time")

        ax.set_ylabel("Variable")
        ax.set_zlabel("Value")

        ax.set_title(f"Batch {batch} 3D Trajectory")

        plt.tight_layout()
        plt.show()      


        from mpl_toolkits.mplot3d import Axes3D



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from mpl_toolkits.mplot3d import Axes3D



class BatchTimeSeriesPlotter:

    def __init__(self, df, time_col="Date and time", batch_col="Batch"):
        
        self.df = df.copy()
        self.time_col = time_col
        self.batch_col = batch_col

        self.df[time_col] = pd.to_datetime(self.df[time_col])

    def _prepare_batch(self, batch, freq):

        df_batch = self.df[self.df[self.batch_col] == batch].copy()

        if df_batch.empty:
            return None, None, None

        df_batch = df_batch.sort_values(self.time_col)

        duplicate_times = df_batch.loc[
            df_batch[self.time_col].duplicated(keep=False),
            self.time_col
        ]

        df_batch = df_batch.groupby(self.time_col).mean(numeric_only=True)

        if df_batch.index.empty:
            return None, None, None

        start = df_batch.index.min()
        end = df_batch.index.max()

        if pd.isna(start) or pd.isna(end):
            return None, None, None

        full_index = pd.date_range(start=start, end=end, freq=freq)

        missing_timestamps = full_index.difference(df_batch.index)

        df_batch = df_batch.reindex(full_index)

        return df_batch, duplicate_times, missing_timestamps

    def plot(self, batch=None, column=None, columns=None, batches=None, freq="15min", width=None):

        if batch is not None and column is not None:
            columns = column

        if isinstance(columns, str):
            columns = [columns]

        if isinstance(column, list):
            columns = column
            column = None

        # MODE 1: batch → multiple columns
        if batch is not None:

            if isinstance(batch, (int,float)):
                batches = [batch]
            else:
                batches = batch

            if columns is None:
                columns = self._get_predictor_columns()

            n = len(columns)

            if width is None:
                width = n

            ncols = min(width, n)
            nrows = math.ceil(n / ncols)

            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols,4*nrows), sharex=True)
            axes = np.atleast_1d(axes).flatten()

            for i, col in enumerate(columns):

                ax = axes[i]

                for b in batches:

                    result = self._prepare_batch(b, freq)
                    if result is None:
                        continue

                    df_batch, _, _ = result

                    if col not in df_batch.columns:
                        continue

                    series = df_batch[col]

                    # normalized time based on full index
                    x = np.linspace(0, 1, len(series))

                    ax.plot(
                        x,
                        series.values,
                        linewidth=1.5,
                        alpha=0.85,
                        label=f"Batch {b}"
                    )

                ax.set_title(col)
                ax.legend()

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle("Batch Comparison")
            plt.tight_layout()
            plt.show()

        # MODE 2: column → multiple batches
        elif column is not None:

            if batches is None:
                batches = sorted(self.df[self.batch_col].dropna().unique())

            if isinstance(batches, (int,float)):
                batches = [batches]

            n = len(batches)

            if width is None:
                width = n

            ncols = min(width, n)
            nrows = math.ceil(n / ncols)

            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols,4*nrows))
            axes = np.atleast_1d(axes).flatten()

            for i, batch in enumerate(batches):

                ax = axes[i]

                df_batch, duplicate_times, missing_timestamps = self._prepare_batch(batch, freq)

                if df_batch is None:
                    continue

                raw_series = df_batch[column]
                interp_series = raw_series.interpolate()

                ax.plot(df_batch.index, raw_series, color="black")

                nan_times = df_batch.index[raw_series.isna()]

                ax.scatter(nan_times, interp_series.loc[nan_times], s=20)
                ax.scatter(missing_timestamps, interp_series.reindex(missing_timestamps), s=25)
                ax.scatter(duplicate_times, interp_series.loc[duplicate_times], s=25)

                ax.set_title(f"Batch {batch}")

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle(column)
            plt.tight_layout()
            plt.show()

        else:
            raise ValueError("Provide either batch=... or column=...")

    def plot_batches_grid(self, columns=None, batches=None, freq="15min"):

        if columns is None:
            columns = self._get_predictor_columns()

        if isinstance(columns, str):
            columns = [columns]

        if batches is None:
            batches = sorted(self.df[self.batch_col].dropna().unique())

        if isinstance(batches, (int, float)):
            batches = [batches]

        nrows = len(columns)
        ncols = len(batches)

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows), sharex=False)
        axes = np.array(axes).reshape(nrows, ncols)

        for col_i, col in enumerate(columns):
            for batch_i, batch in enumerate(batches):

                ax = axes[col_i, batch_i]

                df_batch, duplicate_times, missing_timestamps = self._prepare_batch(batch, freq)

                if df_batch is None:
                    continue

                raw_series = df_batch[col]
                interp_series = raw_series.interpolate()

                ax.plot(df_batch.index, raw_series, linewidth=1)

                nan_times = df_batch.index[raw_series.isna()]

                ax.scatter(nan_times, interp_series.loc[nan_times], s=15)
                ax.scatter(missing_timestamps, interp_series.reindex(missing_timestamps), s=20)
                ax.scatter(duplicate_times, interp_series.loc[duplicate_times], s=20)

                if col_i == 0:
                    ax.set_title(f"Batch {batch}")

                if batch_i == 0:
                    ax.set_ylabel(col)

        plt.tight_layout()
        plt.show()

    def plot_overlay(self, column, batches=None, freq="15min", normalize_time=True):

        if batches is None:
            batches = sorted(self.df[self.batch_col].dropna().unique())

        if isinstance(batches, (int, float)):
            batches = [batches]

        plt.figure(figsize=(10,5))

        for batch in batches:

            result = self._prepare_batch(batch, freq)
            if result is None:
                continue

            df_batch, _, _ = result

            if df_batch is None or df_batch.empty:
                continue

            if column not in df_batch.columns:
                continue

            series = df_batch[column]

            if normalize_time:
                x = np.linspace(0, 1, len(series))
            else:
                x = series.index

            plt.plot(
                x,
                series.values,
                linewidth=1.5,
                alpha=0.85,
                label=f"Batch {batch}"
            )

        plt.xlabel("Normalized Batch Progress" if normalize_time else "Time")
        plt.ylabel(column)
        plt.title(column)

        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_batch_3d(self, batch, columns, freq="15min", normalize_time=True):

        if isinstance(columns, str):
            columns = [columns]

        df_batch, _, _ = self._prepare_batch(batch, freq)

        if df_batch is None or df_batch.empty:
            print("No data for batch")
            return

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')

        for i, col in enumerate(columns):

            if col not in df_batch.columns:
                continue

            series = df_batch[col]

            if normalize_time:
                x = np.linspace(0,1,len(series))
            else:
                x = series.index

            y = np.full(len(series), i)
            z = series.values

            ax.plot(x, y, z, linewidth=2)

        ax.set_yticks(range(len(columns)))
        ax.set_yticklabels(columns)

        ax.set_xlabel("Normalized Batch Progress" if normalize_time else "Time")
        ax.set_ylabel("Variable")
        ax.set_zlabel("Value")

        ax.set_title(f"Batch {batch} 3D Trajectory")

        plt.tight_layout()
        plt.show()
    
    def _get_predictor_columns(self):
        return [
            col for col in self.df.columns
            if col not in [self.time_col, self.batch_col]
            and pd.api.types.is_numeric_dtype(self.df[col])
        ]