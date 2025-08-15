"""
Training script for the All‑NBA prediction models.

This module reads the processed master dataset, performs a rolling
time‑series split (training on seasons up to N‑1 and testing on season N),
and fits two models:

* A linear booster using XGBoost to approximate logistic regression.
* A gradient boosted tree model (XGBoost) tuned for class imbalance.

The predictions and per‑year metrics are saved to ``reports/`` for downstream
analysis.  Computation of precision, recall and F1 are implemented locally
to avoid an explicit dependency on scikit‑learn.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Return counts of TP, FP, TN, FN."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute precision, recall and F1 score."""
    cm = compute_confusion_matrix(y_true, y_pred)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Select numeric features for modeling.

    Exclude non‑numeric columns and target labels.  Returns the filtered
    DataFrame and the list of feature column names.
    """
    ignore = set([
        "name",
        "team",
        "position",
        "season_end_year",
        "was_all_nba",
        "target_all_nba_next",
        "target",
    ])
    numeric_cols = []
    for col, dtype in df.dtypes.items():
        if col in ignore:
            continue
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
    features = df[numeric_cols].fillna(0)
    return features, numeric_cols


def train_and_evaluate(
    master: pd.DataFrame,
    start_year: int,
    min_train_year: int = 2010,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, List[float]]]:
    """Perform rolling evaluation across seasons.

    Parameters
    ----------
    master : pandas.DataFrame
        Processed master table including features and targets.
    start_year : int
        The first season to evaluate (inclusive).
    min_train_year : int, optional
        Minimum year from which to start testing.  Training data will include
        all seasons strictly before the test year.

    Returns
    -------
    metrics : dict
        Nested dictionary keyed by model name then by test_year containing
        precision, recall and F1.
    proba_dict : dict
        Dictionary keyed by model name containing lists of predicted
        probabilities (same order as the test sets concatenated across years).
    """
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {"linear": {}, "xgboost": {}}
    proba_dict: Dict[str, List[float]] = {"linear": [], "xgboost": []}
    # Determine candidate test years
    years = sorted(master["season_end_year"].unique())
    test_years = [y for y in years if y >= min_train_year]
    features, feature_cols = prepare_features(master)
    labels = master["target_all_nba_next"].values
    for year in test_years:
        train_idx = master["season_end_year"] < year
        test_idx = master["season_end_year"] == year
        if train_idx.sum() == 0 or test_idx.sum() == 0:
            continue
        X_train = features.loc[train_idx, feature_cols].values
        y_train = labels[train_idx]
        X_test = features.loc[test_idx, feature_cols].values
        y_test = labels[test_idx]
        # Compute class imbalance ratio for XGBoost
        pos = y_train.sum()
        neg = len(y_train) - pos
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        # Linear model (approximate logistic regression) using gblinear booster
        linear_model = xgb.XGBClassifier(
            objective="binary:logistic",
            booster="gblinear",
            learning_rate=1.0,
            n_estimators=1,
            reg_alpha=0.0,
            reg_lambda=1.0,
            n_jobs=4,
        )
        linear_model.fit(X_train, y_train)
        proba_linear = linear_model.predict_proba(X_test)[:, 1]
        preds_linear = (proba_linear >= 0.5).astype(int)
        proba_dict["linear"].extend(proba_linear.tolist())
        metrics["linear"][str(year)] = compute_metrics(y_test, preds_linear)
        # Gradient boosted model
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            booster="gbtree",
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=4,
        )
        xgb_model.fit(X_train, y_train)
        proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        preds_xgb = (proba_xgb >= 0.5).astype(int)
        proba_dict["xgboost"].extend(proba_xgb.tolist())
        metrics["xgboost"][str(year)] = compute_metrics(y_test, preds_xgb)
    return metrics, proba_dict


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    master_path = data_dir / "master.csv"
    if not master_path.exists():
        raise RuntimeError(f"Master dataset not found at {master_path}")
    master = pd.read_csv(master_path)
    metrics, proba_dict = train_and_evaluate(master, start_year=args.start_year)
    # Persist metrics to JSON
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    # Save predicted probabilities for future stars page
    for model_name, probs in proba_dict.items():
        out_path = reports_dir / f"preds_{model_name}.csv"
        # Concatenate corresponding test rows
        # We'll include year, player and probability for convenience
        test_rows = master[master["season_end_year"] >= args.start_year][[
            "season_end_year",
            "name",
            "team",
        ]].reset_index(drop=True)
        test_rows["probability"] = probs
        test_rows.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train All‑NBA prediction models")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed master dataset",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory to write evaluation reports and predictions",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="First season_end_year to include in testing",
    )
    args = parser.parse_args()
    main(args)
