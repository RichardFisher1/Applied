"""
train.py
========

Entry point for running the full data processing and modelling
pipeline. When executed, this script will:

1. Load and clean the raw operating and product datasets.
2. Compute summary statistics per batch to form the feature matrix.
3. Compute the product rate per batch to form the target vector.
4. Evaluate a suite of machine learning models (linear regression,
   Bayesian ridge and random forest) using k‑fold cross‑validation.
5. Train the selected model on all available batches and predict the
   product rate for the missing batch.
6. Output feature importances (for the random forest) and prediction
   results to the console.

Usage::

    python -m biotech_case_study.src.train

or, from the project root:

    python src/train.py

Author: Applied Bioinformatics case study team, 2026
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Tuple

import numpy as np
import pandas as pd

try:
    # When run as part of the biotech_case_study package
    from .data_processing import build_features_and_target, load_operating_data
    from .models import run_experiments, predict_missing_batch
except ImportError:
    # Fall back to adjusting sys.path when executed as a script
    import os
    import importlib
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    sys.path.insert(0, parent_dir)
    data_processing = importlib.import_module('data_processing')
    models = importlib.import_module('models')
    build_features_and_target = data_processing.build_features_and_target
    load_operating_data = data_processing.load_operating_data
    run_experiments = models.run_experiments
    predict_missing_batch = models.predict_missing_batch


def main(args: argparse.Namespace) -> None:
    # Resolve paths
    operating_csv = pathlib.Path(args.operating).resolve()
    product_xlsx = pathlib.Path(args.product).resolve()
    if not operating_csv.exists() or not product_xlsx.exists():
        sys.stderr.write(f"Input files not found: {operating_csv}, {product_xlsx}\n")
        sys.exit(1)
    # Build dataset
    features, target = build_features_and_target(str(operating_csv), str(product_xlsx), stats=('mean', 'std'))
    # Drop batches with missing target (i.e., the batch without product measurements)
    train_mask = target.notna()
    X_train = features.loc[train_mask]
    y_train = target.loc[train_mask]
    # Run experiments
    print("Evaluating candidate models via cross‑validation...\n")
    results = run_experiments(X_train, y_train, cv=args.cv)
    for res in results:
        print(f"Model: {res.name}")
        print(f"  R² (mean ± std): {res.mean_r2:.3f} ± {res.std_r2:.3f}")
        print(f"  RMSE (mean ± std): {res.mean_rmse:.3f} ± {res.std_rmse:.3f}\n")
    # Identify best model by highest mean R²
    best_model_name = max(results, key=lambda r: r.mean_r2).name
    print(f"Best model based on mean R²: {best_model_name}\n")
    # Load full operating data to identify missing batch
    full_operating_df = load_operating_data(str(operating_csv))
    # Predict missing batch using the best model
    importances, predicted_rate = predict_missing_batch(X_train, y_train, full_operating_df, stats=('mean', 'std'), model_name=best_model_name)
    print(f"Predicted product rate for missing batch: {predicted_rate:.3f} kg/hr\n")
    if importances is not None:
        top_importances = importances.head(args.top_features)
        print("Top feature importances (Random Forest):")
        for feat, imp in top_importances.items():
            print(f"  {feat}: {imp:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run data processing and modelling pipeline for the biotech case study.")
    # By default, look for the data directory under the package root
    root = pathlib.Path(__file__).resolve().parents[1]
    parser.add_argument('--operating', type=str, default=str(root / 'data' / '4000_series_operating_data.csv'), help='Path to the operating CSV file')
    parser.add_argument('--product', type=str, default=str(root / 'data' / '4000_series_product_data.xlsx'), help='Path to the product Excel file')
    parser.add_argument('--cv', type=int, default=5, help='Number of folds for cross‑validation')
    parser.add_argument('--top-features', type=int, default=10, help='Number of top features to display for the random forest')
    args = parser.parse_args()
    main(args)