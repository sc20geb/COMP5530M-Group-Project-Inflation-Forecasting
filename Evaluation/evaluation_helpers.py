import torch
import numpy as np
import os
from glob import glob

def make_evaluation_predictions(model: torch.nn.Module, savepath: str, val_loader: torch.utils.data.DataLoader, device=None, y_scaler=None):
    """
    Loads the model at the savepath specified, and evaluates it using the data held in val_loader.

    Parameters:
    -----------
    model: torch.nn.Module
        Module object into which to load the weights at the specified savepath.
    savepath: str
        Defines the location of the file containing the weights of the model to be loaded.
    val_loader: torch.utils.data.DataLoader
        The DataLoader that provides the evaluation data.
    device: str (optional)
        Idenfities the device to be used when evaluating the model. If not provided, this is identified automatically.
    y_scaler: sklearn.preprocessing.Scaler (optional)
        Scaler with which to inverse transform the predictions made by the model and its actual values.


    Returns:
    --------
    np.array, np.array: Model evaluation predictions and the actual values respectively
    """
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(savepath))
    model = model.to(device)
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())

    # **Concatenate batches into single arrays**
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    # Inverse transform if necessary
    if y_scaler:
        predictions = y_scaler.inverse_transform(predictions)
        actuals = y_scaler.inverse_transform(actuals)

    return predictions, actuals

def get_best_path(savepath : str, model_name : str, stopped_at=-1) -> str:
    """
    Gets the filepaths of the best models with a given name

    Parameters:
    -----------
    savepath: String defining the directory in which model weights are stored
    model_name: String defining the model's name (used to name its weight file)
    stopped_at (optional): Integer defining the epoch at which the model was stopped for fetching specific early-stopped models

    Returns:
    --------
    List of strings defining the file path to the first best model
    """
    stopped = '*'
    if stopped_at != -1: stopped = stopped_at
    return glob(os.path.join(savepath, f"{model_name}_BEST_STOPPED_AT_{stopped}.pth"))

def get_best_model_path(savepath, model_name, verbose=False):
    """
    Gets the filepath of the 'best model available': i.e., either the latest early-stopped model, or the most recent.

    Parameters:
    -----------
    savepath: String defining the directory in which model weights are stored.
    model_name: String defining the model's name (used to name its weight file).
    verbose: Boolean defining whether or not to print out which type of model is used (either early-stopped, or most recent).

    Returns:
    --------
    List of strings defining the file path to the first best model
    """
    # **Find the best saved model dynamically**
    best_model_files = get_best_path(savepath, model_name)
    latest_model_path = os.path.join(savepath, f"{model_name}_latest.pth")

    if best_model_files:
        best_model_path = sorted(best_model_files)[-1]  # Pick the latest best-stopped model
        if verbose: print(f"Best model at: {best_model_path}")
    elif os.path.exists(latest_model_path):
        best_model_path = latest_model_path
        if verbose: print(f"No early-stopped model found. Best model is instead the latest at: {best_model_path}")
    else:
        raise FileNotFoundError("No saved model found! Ensure training was completed successfully.")
    
    return best_model_path