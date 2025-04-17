
# hyperparameters.py

import optuna
import json
import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Define GRU search space
OPTUNA_SEARCH_SPACE = {
    "hidden_size": (32, 256),
    "num_layers": (1, 4),
    "lr": (1e-5, 1e-1),
}

BEST_HYPERPARAMETERS_FILE = "best_hyperparameters.json"  # Store best params
BEST_HYPERPARAMETERS = None  # Cached best params

def save_best_hyperparameters(best_params, verbose=False):
    """Save best hyperparameters to a JSON file."""
    with open(BEST_HYPERPARAMETERS_FILE, "w") as f:
        json.dump(best_params, f)
    if verbose: print(" Best hyperparameters saved!")

def load_best_hyperparameters(verbose=False):
    """Load best hyperparameters from file."""
    global BEST_HYPERPARAMETERS
    if os.path.exists(BEST_HYPERPARAMETERS_FILE):
        with open(BEST_HYPERPARAMETERS_FILE, "r") as f:
            BEST_HYPERPARAMETERS = json.load(f)
        if verbose: print(" Loaded best hyperparameters:", BEST_HYPERPARAMETERS)
    return BEST_HYPERPARAMETERS

def tune_hyperparameters(objective, n_trials=20):
    """Runs Optuna hyperparameter tuning and stores the best parameters."""
    global BEST_HYPERPARAMETERS
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    BEST_HYPERPARAMETERS = study.best_params
    save_best_hyperparameters(BEST_HYPERPARAMETERS)
    return BEST_HYPERPARAMETERS

# Random Forest Hyperparameter block

'''

# ---------------- new Random‑Forest extension (non‑breaking with GRU) -----------------

RF_SEARCH_SPACE = {
    "n_estimators": (200, 1000),
    "max_depth": (3, 20),  
    "min_samples_leaf": (1, 15),
    "max_features": ("sqrt", "log2", 0.3, 0.5, None),
}


def _suggest_rf(trial: optuna.Trial):
    """Helper to draw a set of RF params from RF_SEARCH_SPACE."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", *RF_SEARCH_SPACE["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *RF_SEARCH_SPACE["max_depth"]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", *RF_SEARCH_SPACE["min_samples_leaf"]),
        "max_features": trial.suggest_categorical("max_features", RF_SEARCH_SPACE["max_features"]),
    }


def tune_rf_pipeline(pipeline, X, y, *, n_trials: int = 40, n_splits: int = 5, random_state: int = 0):
    """Time‑series aware Optuna tuning for a *Pipeline* ending with RandomForestRegressor.

    The function **does not** touch any of the original GRU utilities; it simply
    re‑uses `save_best_hyperparameters` to persist the best RF params alongside
    any previously tuned models.
    """
    def objective(trial: optuna.Trial):
        params = _suggest_rf(trial)
        pipeline.set_params(**{f"rf__{k}": v for k, v in params.items()})
        cv = TimeSeriesSplit(n_splits=n_splits)
        rmses = []
        for train_idx, val_idx in cv.split(X):
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = pipeline.predict(X.iloc[val_idx])
            rmses.append(mean_squared_error(y.iloc[val_idx], pred, squared=False))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Persist and return best params (non‑destructive to other models)
    save_best_hyperparameters(study.best_params)
    return study.best_params

'''