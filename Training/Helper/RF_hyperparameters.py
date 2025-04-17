# RF_hyperparameters.py – stand‑alone Optuna tuner for Random‑Forest pipelines
"""Minimal, *self‑contained* helper so Random‑Forest hyper‑parameter tuning
can live next to the original GRU tuner without touching it.

Public API
----------
>>> from Training.Helper.RF_hyperparameters import tune_rf
>>> best_params = tune_rf(rf_pipeline, X_train, y_train, n_trials=50)

The function expects an **sklearn Pipeline** whose final step is a
`RandomForestRegressor`; it performs leakage‑free cross‑validation via
`TimeSeriesSplit`, safeguards against non‑finite predictions, caches the best
parameters in `best_rf_hyperparameters.json`, and returns them as a dict.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------

RF_SEARCH_SPACE = {
    "n_estimators": (200, 1000),          # inclusive ints
    "max_depth": (3, 20),                # inclusive ints, None added below
    "min_samples_leaf": (1, 15),         # inclusive ints
    "max_features": ["sqrt", "log2", 0.3, 0.5, None],
}

BEST_PARAMS_FILE = Path(__file__).with_name("best_rf_hyperparameters.json")


def _suggest_rf(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Sample a set of RF hyper‑parameters from the search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", *RF_SEARCH_SPACE["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *RF_SEARCH_SPACE["max_depth"]),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", *RF_SEARCH_SPACE["min_samples_leaf"]
        ),
        "max_features": trial.suggest_categorical(
            "max_features", RF_SEARCH_SPACE["max_features"]
        ),
    }


def _save_best(params: Dict[str, Any]) -> None:
    with BEST_PARAMS_FILE.open("w") as fh:
        json.dump(params, fh)


def _load_best() -> Optional[Dict[str, Any]]:
    if BEST_PARAMS_FILE.exists():
        with BEST_PARAMS_FILE.open() as fh:
            return json.load(fh)
    return None

# -----------------------------------------------------------------------------
# Tuner (public)
# -----------------------------------------------------------------------------

def tune_rf(
    pipeline: Pipeline,
    X,
    y,
    *,
    n_trials: int = 40,
    cv_splits: int = 5,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """Tune the RF step of *pipeline* using Optuna.

    The pipeline is modified *in‑place* so the caller can directly refit it after
    tuning.
    """

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_rf(trial)
        pipeline.set_params(**{f"rf__{k}": v for k, v in params.items()})

        tscv = TimeSeriesSplit(n_splits=cv_splits)
        rmses = []
        for train_idx, val_idx in tscv.split(X):
            try:
                pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
                pred = pipeline.predict(X.iloc[val_idx])
                if not np.all(np.isfinite(pred)):
                    raise ValueError("non‑finite predictions")
                rmse = mean_squared_error(y.iloc[val_idx], pred, squared=False)
            except Exception:
                # any failure → penalise trial
                return float("inf")
            rmses.append(rmse)
        return float(np.mean(rmses))

    # ------------------------------------------------------------------
    # Run Optuna
    # ------------------------------------------------------------------

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    best_params_ordered = study.best_trial.params  # keep Optuna's original order
    # Apply best params to pipeline
    pipeline.set_params(**{f"rf__{k}": v for k, v in best_params_ordered.items()})

    # Cache for future runs
    _save_best(best_params_ordered)

    return best_params_ordered