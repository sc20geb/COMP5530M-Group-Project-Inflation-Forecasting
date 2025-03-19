
# hyperparameters.py

import optuna
import json
import os

# Define search space
OPTUNA_SEARCH_SPACE = {
    "hidden_size": (32, 256),
    "num_layers": (1, 4),
    "lr": (1e-5, 1e-1),
}

BEST_HYPERPARAMETERS_FILE = "best_hyperparameters.json"  # Store best params
BEST_HYPERPARAMETERS = None  # Cached best params

def save_best_hyperparameters(best_params):
    """Save best hyperparameters to a JSON file."""
    with open(BEST_HYPERPARAMETERS_FILE, "w") as f:
        json.dump(best_params, f)
    print(" Best hyperparameters saved!")

def load_best_hyperparameters():
    """Load best hyperparameters from file."""
    global BEST_HYPERPARAMETERS
    if os.path.exists(BEST_HYPERPARAMETERS_FILE):
        with open(BEST_HYPERPARAMETERS_FILE, "r") as f:
            BEST_HYPERPARAMETERS = json.load(f)
        print(" Loaded best hyperparameters:", BEST_HYPERPARAMETERS)
    return BEST_HYPERPARAMETERS

def tune_hyperparameters(objective, n_trials=20):
    """Runs Optuna hyperparameter tuning and stores the best parameters."""
    global BEST_HYPERPARAMETERS
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    BEST_HYPERPARAMETERS = study.best_params
    save_best_hyperparameters(BEST_HYPERPARAMETERS)
    return BEST_HYPERPARAMETERS
