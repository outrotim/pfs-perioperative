#!/usr/bin/env python3
"""Reproduce aggregate Study 22 internal-validation results.

The script requires a locally authorised analytic table. It deliberately
writes aggregate JSON only and never exports identifiers, fold assignments,
individual outcomes, or individual predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260721
GROUPED_SEED = 20260722
OUTCOME = "outcome_composite"
GROUP = "subjectid"
MODELS = {
    "pfs5": ["age", "asa", "hrv_sdnn", "map_successive_var", "ncc_index"],
    "age_asa": ["age", "asa"],
    "physiology3": ["hrv_sdnn", "map_successive_var", "ncc_index"],
    "asa_only": ["asa"],
}


def make_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                IterativeImputer(
                    random_state=seed,
                    max_iter=20,
                    sample_posterior=False,
                    skip_complete=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight=None,
                    random_state=seed,
                ),
            ),
        ]
    )


def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = np.finfo(float).eps
    clipped = np.clip(p, eps, 1 - eps)
    logit = np.log(clipped / (1 - clipped))
    fit = sm.Logit(y, sm.add_constant(logit)).fit(disp=False)
    return float(fit.params[0]), float(fit.params[1])


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    intercept, slope = calibration(y, p)
    return {
        "auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def case_level_folds(data: pd.DataFrame) -> np.ndarray:
    y = data[OUTCOME].astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    folds = np.full(len(data), -1, dtype=int)
    for fold, (_, test) in enumerate(splitter.split(data, y)):
        folds[test] = fold
    return folds


def patient_grouped_folds(data: pd.DataFrame) -> np.ndarray:
    if GROUP not in data:
        raise ValueError("Patient-grouped mode requires a subjectid column")
    y = data[OUTCOME].astype(int).to_numpy()
    groups = data[GROUP].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=10, shuffle=True, random_state=GROUPED_SEED
    )
    folds = np.full(len(data), -1, dtype=int)
    for fold, (_, test) in enumerate(splitter.split(data, y, groups)):
        folds[test] = fold
    patient_fold_count = (
        pd.DataFrame({"group": groups, "fold": folds})
        .groupby("group")["fold"]
        .nunique()
    )
    if int(patient_fold_count.max()) != 1:
        raise RuntimeError("A patient was assigned to more than one fold")
    return folds


def oof_predictions(
    data: pd.DataFrame, predictors: list[str], folds: np.ndarray, seed: int
) -> np.ndarray:
    y = data[OUTCOME].astype(int).to_numpy()
    oof = np.full(len(data), np.nan)
    for fold in range(10):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        fit = make_pipeline(seed + fold + 1)
        fit.fit(data.iloc[train][predictors], y[train])
        oof[test] = fit.predict_proba(data.iloc[test][predictors])[:, 1]
    if np.isnan(oof).any():
        raise RuntimeError("Incomplete out-of-fold predictions")
    return oof


def bootstrap_indices(
    data: pd.DataFrame, grouped: bool, replicates: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    y = data[OUTCOME].astype(int).to_numpy()
    samples = []
    if grouped:
        groups = data[GROUP].drop_duplicates().to_numpy()
        group_rows = {
            group: np.flatnonzero(data[GROUP].to_numpy() == group) for group in groups
        }
    while len(samples) < replicates:
        if grouped:
            selected = rng.choice(groups, size=len(groups), replace=True)
            idx = np.concatenate([group_rows[group] for group in selected])
        else:
            idx = rng.integers(0, len(data), len(data))
        if np.unique(y[idx]).size == 2:
            samples.append(idx)
    return samples


def interval(values: list[float]) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def evaluate(data: pd.DataFrame, grouped: bool, replicates: int) -> dict:
    required = {OUTCOME, *(name for values in MODELS.values() for name in values)}
    if grouped:
        required.add(GROUP)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    y = data[OUTCOME].astype(int).to_numpy()
    if np.unique(y).size != 2:
        raise ValueError("Outcome must contain both classes")
    folds = patient_grouped_folds(data) if grouped else case_level_folds(data)
    seed = GROUPED_SEED if grouped else SEED
    predictions = {
        name: oof_predictions(data, predictors, folds, seed)
        for name, predictors in MODELS.items()
    }
    point = {name: metrics(y, p) for name, p in predictions.items()}
    samples = bootstrap_indices(data, grouped, replicates, seed)
    pfs_auc = [roc_auc_score(y[idx], predictions["pfs5"][idx]) for idx in samples]
    pfs_ap = [
        average_precision_score(y[idx], predictions["pfs5"][idx]) for idx in samples
    ]
    pfs_brier = [
        brier_score_loss(y[idx], predictions["pfs5"][idx]) for idx in samples
    ]
    delta = [
        roc_auc_score(y[idx], predictions["pfs5"][idx])
        - roc_auc_score(y[idx], predictions["age_asa"][idx])
        for idx in samples
    ]
    return {
        "privacy": "aggregate output only; no row-level predictions are written",
        "mode": "patient_grouped" if grouped else "case_level",
        "cohort": {
            "cases": int(len(data)),
            "patients": int(data[GROUP].nunique()) if GROUP in data else None,
            "events": int(y.sum()),
        },
        "validation": {
            "folds": 10,
            "fold_contained_imputation_and_scaling": True,
            "bootstrap_replicates": replicates,
        },
        "models": point,
        "pfs5_ci_95": {
            "auc": interval(pfs_auc),
            "average_precision": interval(pfs_ap),
            "brier": interval(pfs_brier),
        },
        "pfs5_minus_age_asa": {
            "delta_auc": point["pfs5"]["auc"] - point["age_asa"]["auc"],
            "ci_95": interval(delta),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("aggregate_evaluation.json"))
    parser.add_argument("--patient-grouped", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be a positive integer")
    data = pd.read_csv(args.input)
    result = evaluate(data, args.patient_grouped, args.bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote aggregate-only results to {args.output}")


if __name__ == "__main__":
    main()
