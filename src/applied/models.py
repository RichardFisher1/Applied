"""
models.py
=========

This module defines a collection of machine learning models and
helpers for training and evaluating them on the summarised batch
features. The models here are intended to be illustrative rather than
exhaustive; they include:

* Linear regression (ordinary least squares) – a baseline model.
* Bayesian Ridge regression – a Bayesian linear model with Gaussian
  priors on the coefficients (sometimes referred to as Bayesian
  linear regression) implemented in scikit‑learn.
* Random Forest regression – an ensemble of decision trees which
  naturally handles non‑linear relationships and interactions. The
  impurity‑based feature importance is extracted to identify which
  process variables contribute most strongly to the target. Note
  however that impurity‑based importances can be biased for
  high‑cardinality features【252901485110863†L573-L626】.

Cross‑validation is used throughout to avoid overfitting: the data
are split into multiple folds; models are trained on training folds
and evaluated on validation folds. Cross‑validation helps ensure
that performance metrics estimate generalisation ability【172510941299323†L125-L135】. A
5‑fold strategy is adopted by default but can be adjusted.

Functions are provided to perform model fitting, compute evaluation
metrics (R² and root‑mean‑squared error) and report cross‑validated
scores. Once a promising model has been identified, it can be
retrained on all training batches and used to predict the product
rate for a previously unseen batch (the 22nd batch). The
``predict_missing_batch`` function demonstrates this workflow.

Author: Applied Bioinformatics case study team, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@dataclass
class ModelResult:
    """Container for cross‑validation results for a single model."""
    name: str
    cv_r2_scores: np.ndarray
    cv_rmse_scores: np.ndarray
    mean_r2: float
    std_r2: float
    mean_rmse: float
    std_rmse: float

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

def bootstrap_r2(y_true, y_pred, n_bootstrap=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    boot_scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        r2_sample = r2_score(y_true.iloc[idx], y_pred[idx])
        boot_scores.append(r2_sample)

    boot_scores = np.array(boot_scores)

    return (
        boot_scores.mean(),
        boot_scores.std(),
        np.percentile(boot_scores, 2.5),
        np.percentile(boot_scores, 97.5),
    )

def correlation_report(X, y=None, threshold=0.75, mode="both"):
    """
    Correlation diagnostics.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series or array-like, optional
        Target variable
    threshold : float
        Minimum absolute correlation to report
    mode : {"feature", "target", "duplicates", "both"}
        feature     → X vs X correlations
        target      → X vs y correlations
        duplicates  → duplicate feature detection
        both        → feature + target correlations

    Returns
    -------
    Depending on mode:
        feature_corr_df
        target_corr_df
        duplicate_features
    """

    import numpy as np
    import pandas as pd

    X = X.copy()

    feature_corr_df = None
    target_corr_df = None
    duplicate_features = None

    # -----------------------------
    # Duplicate features
    # -----------------------------
    if mode in ["duplicates", "both"]:
        duplicate_cols = X.T.duplicated()
        duplicate_features = X.columns[duplicate_cols].tolist()

    # -----------------------------
    # Feature vs Feature
    # -----------------------------
    if mode in ["feature", "both"]:

        corr_matrix = X.corr().abs()

        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        feature_pairs = [
            (idx, col, upper_triangle.loc[idx, col])
            for col in upper_triangle.columns
            for idx in upper_triangle.index
            if upper_triangle.loc[idx, col] > threshold
        ]

        feature_corr_df = (
            pd.DataFrame(feature_pairs, columns=["Feature_1", "Feature_2", "Correlation"])
            .sort_values("Correlation", ascending=False)
            .reset_index(drop=True)
        )

    # -----------------------------
    # Feature vs Target
    # -----------------------------
    if y is not None and mode in ["target", "both"]:

        target_corr = X.corrwith(y).abs()

        target_corr_df = (
            target_corr[target_corr > threshold]
            .sort_values(ascending=False)
            .reset_index()
        )

        target_corr_df.columns = ["Feature", "Correlation_with_Target"]

    # -----------------------------
    # Return depending on mode
    # -----------------------------
    if mode == "feature":
        return feature_corr_df

    if mode == "target":
        return target_corr_df

    if mode == "duplicates":
        return duplicate_features

    return feature_corr_df, target_corr_df, duplicate_features







def evaluate_models(X, y, n_bootstrap=1000):

    # ----------------------------
    # Align & clean
    # ----------------------------
    y = y.reindex(X.index)

    mask = y.notna()
    X_model = X.loc[mask]
    y_model = y.loc[mask]

    # ----------------------------
    # Define models
    # ----------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "PLS (2 comps)": PLSRegression(n_components=2),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42
        )
    }

    loo = LeaveOneOut()
    results = []

    # ----------------------------
    # Evaluation loop
    # ----------------------------
    for name, model in models.items():

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        # MAE
        scores_mae = cross_val_score(
            pipe,
            X_model,
            y_model,
            cv=loo,
            scoring="neg_mean_absolute_error"
        )

        mae = -scores_mae.mean()

        # LOOCV predictions
        preds = cross_val_predict(pipe, X_model, y_model, cv=loo)

        r2 = r2_score(y_model, preds)
        rmse = np.sqrt(mean_squared_error(y_model, preds))

        # ----------------------------
        # Bootstrap R²
        # ----------------------------
        boot_mean, boot_std, ci_low, ci_high = bootstrap_r2(
            y_model, preds, n_bootstrap=n_bootstrap
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
            "Relative MAE (%)": 100 * mae / y_model.mean(),
            "Bootstrap R² Mean": boot_mean,
            "Bootstrap R² Std": boot_std,
            "R² 95% CI Lower": ci_low,
            "R² 95% CI Upper": ci_high
        })

    results_df = pd.DataFrame(results).sort_values("R²", ascending=False)

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    return results_df

def evaluate_models3(X, y):

    # ----------------------------
    # Align & clean
    # ----------------------------
    y = y.reindex(X.index)

    mask = y.notna()
    X_model = X.loc[mask]
    y_model = y.loc[mask]

    print("Number of batches:", len(y_model))
    print("Mean productivity:", y_model.mean())
    print("Std productivity:", y_model.std())

    # ----------------------------
    # Define models
    # ----------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "PLS (2 comps)": PLSRegression(n_components=2),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42
        )
    }

    loo = LeaveOneOut()
    results = []

    # ----------------------------
    # Evaluation loop
    # ----------------------------
    for name, model in models.items():

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        # MAE
        scores_mae = cross_val_score(
            pipe,
            X_model,
            y_model,
            cv=loo,
            scoring="neg_mean_absolute_error"
        )

        mae = -scores_mae.mean()

        # R² and RMSE
        preds = cross_val_predict(pipe, X_model, y_model, cv=loo)

        r2 = r2_score(y_model, preds)
        rmse = np.sqrt(mean_squared_error(y_model, preds))

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
            "Relative MAE (%)": 100 * mae / y_model.mean()
        })

    results_df = pd.DataFrame(results).sort_values("R²", ascending=False)

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    return results_df

def evaluate_model1(
    model, X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> ModelResult:
    """Evaluate a regression model via cross‑validation.

    Parameters
    ----------
    model : scikit‑learn estimator
        The model to evaluate. Should implement ``fit`` and
        ``predict`` methods.
    X : pd.DataFrame
        Feature matrix indexed by batch.
    y : pd.Series
        Target vector indexed by batch.
    cv : int, optional
        Number of folds for cross‑validation. Must be at least 2.

    Returns
    -------
    ModelResult
        An object containing per‑fold and aggregated metrics.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    # Use cross_val_score to compute R²; returns array of scores
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    # For RMSE we compute manually: cross_val_score returns negative
    # mean squared error by convention; we take square root of
    # negatives
    neg_mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-neg_mse_scores)
    return ModelResult(
        name=model.__class__.__name__,
        cv_r2_scores=r2_scores,
        cv_rmse_scores=rmse_scores,
        mean_r2=np.mean(r2_scores),
        std_r2=np.std(r2_scores),
        mean_rmse=np.mean(rmse_scores),
        std_rmse=np.std(rmse_scores),
    )

def get_models(n_estimators: int = 500, random_state: int = 42) -> Dict[str, object]:
    """Construct a dictionary of candidate regression models.

    This allows centralised definition of the models used in the
    experiments. A standard scaling step is applied before linear
    models to ensure that features are on comparable scales【172510941299323†L187-L205】.

    Parameters
    ----------
    n_estimators : int, optional
        Number of trees in the random forest. Defaults to 500.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    Dict[str, object]
        Mapping from model name to a configured estimator.
    """
    models: Dict[str, object] = {}
    from sklearn.impute import SimpleImputer
    # Simple imputer that replaces missing values with the mean of each column
    imputer = SimpleImputer(strategy='mean')
    # Ordinary least squares with imputation and scaling
    models['LinearRegression'] = Pipeline([
        ('imputer', imputer),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    # Bayesian Ridge regression: a Bayesian linear model with Gaussian priors on coefficients
    models['BayesianRidge'] = Pipeline([
        ('imputer', imputer),
        ('scaler', StandardScaler()),
        ('regressor', BayesianRidge(compute_score=True))
    ])
    # Random Forest regression: ensemble of decision trees with imputation
    models['RandomForest'] = Pipeline([
        ('imputer', imputer),
        ('regressor', RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            oob_score=False
        ))
    ])
    return models

def run_experiments(
    X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> List[ModelResult]:
    """Train and evaluate a suite of models.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    cv : int, optional
        Number of cross‑validation folds. Defaults to 5.

    Returns
    -------
    List[ModelResult]
        List of ModelResult objects summarising performance.
    """
    results: List[ModelResult] = []
    for name, model in get_models().items():
        res = evaluate_model(model, X, y, cv=cv)
        # Overwrite model name (pipeline's class is Pipeline)
        res.name = name
        results.append(res)
    return results

def compute_feature_importances(
    rf_model: RandomForestRegressor, feature_names: List[str]
) -> pd.Series:
    """Extract and sort feature importances from a trained random forest.

    Parameters
    ----------
    rf_model : RandomForestRegressor
        A fitted random forest regressor.
    feature_names : list of str
        Names of features corresponding to the training data used to
        fit the model.

    Returns
    -------
    pd.Series
        Series of importances indexed by feature name, sorted in
        descending order.

    Notes
    -----
    The feature importance scores are based on the mean decrease in
    impurity (MDI) across the ensemble【252901485110863†L573-L626】. While
    informative, they can be biased toward high‑cardinality features.
    Consider permutation importance for a more robust measure.
    """
    importances = rf_model.feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)

def predict_missing_batch(
    X: pd.DataFrame,
    y: pd.Series,
    full_operating_df: pd.DataFrame,
    stats: Tuple[str, ...] = ("mean", "std"),
    model_name: str = 'RandomForest'
) -> Tuple[pd.DataFrame, float]:
    """Train a model on the available batches and predict for the missing one.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix for the training batches (index is Batch).
    y : pd.Series
        Target vector for the training batches.
    full_operating_df : pd.DataFrame
        The full cleaned operating data (including the batch without
        product measurements).
    stats : tuple of str, optional
        Summary statistics used when computing features. Should match
        the statistics used for ``X``.
    model_name : str, optional
        Name of the model to use for prediction (must exist in
        ``get_models()``). Defaults to ``RandomForest``.

    Returns
    -------
    Tuple[pd.DataFrame, float]
        A tuple containing:
        * The per‑feature importances as a Series if the model is a
          RandomForest, otherwise ``None``.
        * The predicted product rate for the missing batch.

    Notes
    -----
    This function identifies the batch present in the operating data
    but not in ``y``, trains the specified model on the available
    batches and predicts the product rate for the missing batch.
    """
    models = get_models()
    if model_name not in models:
        raise ValueError(f"Unknown model {model_name}. Choose from {list(models.keys())}.")
    model = models[model_name]
    # Fit on all available data
    model.fit(X, y)
    # Determine the missing batch: those present in operating data but not in target
    all_batches = set(full_operating_df['Batch'].unique())
    train_batches = set(y.index.tolist())
    missing_batches = list(all_batches - train_batches)
    if not missing_batches:
        raise ValueError("No missing batch to predict.")
    missing_batch = missing_batches[0]
    # Compute features for the missing batch using the same summary function
    # Import summarise_batches lazily to avoid circular imports
    try:
        from .data_processing import summarise_batches
    except ImportError:
        # Fallback if relative import fails
        import importlib
        summarise_batches = importlib.import_module('data_processing').summarise_batches
    summary_df = summarise_batches(full_operating_df, stats=stats)
    # Extract the row corresponding to the missing batch
    missing_row = summary_df[summary_df['Batch'] == missing_batch].drop(columns=['Batch'])
    # Ensure the feature order matches training features and fill missing columns with NaN
    missing_features = missing_row.reindex(columns=X.columns, fill_value=np.nan)
    missing_features.index = [missing_batch]
    # Predict
    predicted_rate = float(model.predict(missing_features)[0])
    # Compute feature importances if RandomForest
    importances = None
    if model_name == 'RandomForest':
        # When using a pipeline, extract the underlying regressor
        reg = None
        if hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
            reg = model.named_steps['regressor']
        elif hasattr(model, 'feature_importances_'):
            reg = model
        if reg is not None and hasattr(reg, 'feature_importances_'):
            importances = compute_feature_importances(reg, list(X.columns))
    return importances, predicted_rate