import torch
import numpy as np
import os

def evaluate_model(model: torch.nn.Module, savepath: str, X_test: np.array, exog_test=None, device=None, y_scaler=None):
    """
    Loads given saved model and evaluates on test set

    Parameters:
    -----------
    model: Instance of model architecture to load weights into and evaluate
    savepath: String defining where the saved model is
    X_test: Test data on which to evaluate the model
    exog_test: Optional exogenous variables for the test set
    device (optional): Torch device to transfer the test data to before running through the model
    y_scaler (optional): Scaler for prediction values

    Returns:
    --------
    Predictions of the loaded model on the test set provided
    """
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(savepath))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Convert exog_test to tensor if needed
    if exog_test is not None:
        exog_test_tensor = torch.tensor(exog_test, dtype=torch.float32).to(device)
        predictions = model(X_test_tensor, exog_test_tensor).detach().cpu().numpy()
    else:
        predictions = model(X_test_tensor).detach().cpu().numpy()


    if y_scaler:
        return y_scaler.inverse_transform(predictions)
    return predictions

def get_best_path(savepath : str, model_name : str, stopped_at=-1) -> str:
    """
    Gets the filepath of the first best model with a given name

    Parameters:
    -----------
    savepath: String defining the directory in which model weights are stored
    model_name: String defining the model's name (used to name its weight file)
    stopped_at (optional): Integer defining the epoch at which the model was stopped for fetching specific early-stopped models

    Returns:
    --------
    String defining the file path to the first best model
    """
    if stopped_at != -1: bestFile = f'{model_name}_BEST_STOPPED_AT_{stopped_at}.pth'
    #Could select any of the items in this list; they each represent a single training run
    else: bestFile = [path for path in os.listdir(savepath) if 'BEST' in path and model_name in path][0]
    return os.path.join(savepath, bestFile)