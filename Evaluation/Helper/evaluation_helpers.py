import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator

def make_evaluation_predictions(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, savepath: str = '', device=None, y_scaler : BaseEstimator = None):
    """
    Loads the model at the savepath specified, and evaluates it using the data held in val_loader.

    Parameters:
    -----------
    model: torch.nn.Module
        Module object into which to load the weights at the specified savepath.
    val_loader: torch.utils.data.DataLoader
        The DataLoader that provides the evaluation data.
    savepath: str (optional)
        Defines the location of the file containing the weights of the model to be loaded. If not provided, it is assumed the model is already trained.
    device: str (optional)
        Idenfities the device to be used when evaluating the model. If not provided, this is identified automatically.
    y_scaler: sklearn BaseEstimator (optional)
        Scaler with which to inverse transform the predictions made by the model and its actual values.


    Returns:
    --------
    np.array, np.array: Model evaluation predictions and the actual values respectively
    """
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if savepath: model.load_state_dict(torch.load(savepath))
    model = model.to(device)
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        # Can capture multiple inputs (e.g., if the model takes exogenous variables)
        for *inputs, targets in val_loader:
            inputs = [input.to(device) for input in inputs]
            targets = targets.to(device)
            outputs = model(*inputs)
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

def evaluate_model(model : torch.nn.Module, val_loader: torch.utils.data.DataLoader, y_scaler : BaseEstimator, observation_dates : list, device : torch.device, 
                   print_dates : int = 10, savepath : str = '', verbose : bool = False, 
                   metrics : dict ={'RMSE': root_mean_squared_error, 
                                    'MAE': mean_absolute_error,
                                    'r2': r2_score}):
    """
    Evaluates a trained model on validation data, plots its predictions, and returns associated metrics.

    Parameters:
    -----------
    model: torch.nn.Module
        The trained PyTorch model to evaluate. (if savepath is passed, weights from that file are loaded into this object)
    val_loader: DataLoader
        DataLoader containing validation data.
    y_scaler: BaseEstimator object
        The scaler used for inverse transforming predictions.
    observation_dates: list or pd.Series
        The dates corresponding to validation predictions.
    device: torch.device
        The device to run predictions on (CPU/GPU).
    print_dates: int
        Minimum number of dates to print on the plot (earliest and latest dates are always included)
    savepath: str (optional)
        Defines the location of the file containing the weights of the model to be loaded. If not provided, it is assumed the model is already trained.
    verbose: boolean (optional)
        Whether or not to print progress updates/ outputs.

    Returns:
    --------
    df_comparison: pd.DataFrame
        DataFrame containing actual vs predicted values.
    rmse: float
        Root Mean Squared Error (RMSE).
    """
    predictions_inv, actuals_inv = make_evaluation_predictions(model, val_loader, savepath=savepath, device=device, y_scaler=y_scaler)

    # Extract the dates for validation predictions
    val_dates = observation_dates[-len(actuals_inv):]

    ax = display_results(actuals_inv.flatten(), predictions_inv.flatten(), val_dates, type(model).__name__, print_dates=print_dates)

    if verbose: plt.show()

    # Create DataFrame in necessary format
    actual_predicted_df = pd.DataFrame(np.concatenate((actuals_inv, predictions_inv), axis=1), columns=['ground_truth', type(model).__name__])
    model_metrics = calc_metrics(actual_predicted_df, metrics=metrics)
    # Compute metrics for validation predictions
    return ax, model_metrics

def display_results(actuals : np.array, predictions : np.array, dates : list, model_name : str, print_dates : int = 10):
    # Plot actual vs predicted Inflation values
    plt.figure(figsize=(12, 6))
    ax = plt.axes()

    plt.plot(dates, actuals, label='Actual Inflation', linestyle='-', linewidth=2)
    plt.plot(dates, predictions, label='Predicted Inflation', linestyle='--', linewidth=2)

    # Ensure correct number of ticks are printed, and that the earliest and latest dates are always printed
    xlimLower, xlimUpper = ax.get_xticks()[0], ax.get_xticks()[-1]
    newTicks = list(np.arange(xlimLower, xlimUpper+1e-5, step=(xlimUpper-xlimLower)/(print_dates-1)))
    ax.set_xticks(newTicks)

    plt.xlabel("Date")
    plt.ylabel("Inflation")
    plt.title(f"Actual vs. {model_name} Predicted PCE")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    return ax

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

def get_model_name(fileName:str):
    '''
    This function gets the model name according to the following naming convention: {model name}_{information...}.npy,
    where {information} is optional. 

    Parameters:
    -----------
    fileName: A string of the models predictions file name.

    Returns:
    --------
    A string of the models name.
    '''

    return fileName.split('_')[0].split('.')[0]


def get_predictions(predsPath:Path):
    '''
    This function combines all predictions of all models into one dataframe and includes the ground truth.

    Parameters:
    -----------
    predsPath: A path/string of a path, to the directory of all the predictions

    Returns:
    --------
    Returns a pandas Dataframe of all the models predictions, as well as the ground truth. 
    '''
    # Get the ground truth:
    groundTruth= pd.read_csv(Path('../Data/Test/test1990s.csv'),parse_dates=[0],date_format='%m%Y',index_col=0, usecols=[0,1])

    # make an empty Dataframe to store the models predictions (making the index  the observation data):
    predsDf= pd.DataFrame(index=groundTruth.index)

    # Add the ground truth to the predictions dataframe
    predsDf['ground_truth']= groundTruth

    # Loop over all the files in the predictions folder:
    for i in list(predsPath.glob('*.npy')):
        # Add the predictions to the predictions dataframe, where the column is the model name
        predsDf[get_model_name(i.name)]= np.load(i)[:12]
    
    return predsDf

def calc_metrics(predictionsDf:pd.DataFrame, horizon = None, metrics={'RMSE': root_mean_squared_error, 
                                                                      'MAE': mean_absolute_error,
                                                                      'r2': r2_score}):
    '''
    This function calculates the evaluation metrics of each model, given the predictions dataframe.
    The following metrics are used by default:
        * RMSE
        * MAE
        * r^2 score

    Parameters:
    -----------
    predictionsDf: a pandas DataFrame containing the ground truth and all the predictions for each model, organized in columns.

    horizon: number of timesteps to calculate the metrics.

    metrics: dict
        Names of metrics to be used in columns and their associated function (must take arguments: y, y_hat)

    Returns:
    --------
    Returns a pandas Dataframe containg all the evaluation metrics of all the models, where each column represents a metric and each row represents a model.
    '''
    # Deafult to horizon of 2 years:
    if horizon is None:
        horizon= predictionsDf.shape[0]

    # create an empty dataframe with columns reprresnting an evaluation metric
    metricsDf= pd.DataFrame(columns=list(metrics.keys()))
    
    # Loop over all columns/models in the prtedictions dataframe
    for model in predictionsDf.columns.drop('ground_truth'):
        # Calculate the metrics and add them to the metrics dataframe
        for metric in metrics:
            metricsDf.loc[model, metric] = metrics[metric](predictionsDf['ground_truth'].iloc[:horizon], predictionsDf[model].iloc[:horizon])

    return metricsDf

def calc_metrics_arrays(*prediction_arrays : np.ndarray, model_names : list[str] = None, **calc_metrics_kwargs):
    '''
    Provides access to calc_metrics by passing a variable number of arrays, and a list of model names.

    Parameters:
    -----------
    prediction_arrays: a variable number of numpy arrays containing the ground truth and all the predictions for each model (in that order).

    model_names (optional): a list of strings, one for each model that has had its predictions passed. If not given, the models are each associated with an integer.

    calc_metrics_kwargs (optional): the keyword arguments for calc_metrics if needed (i.e. horizon : int and metrics : dict)

    Returns:
    --------
    Returns a pandas Dataframe containg all the evaluation metrics of all the models, where each column represents a metric and each row represents a model.
    '''
    predictionsDf = pd.DataFrame(np.hstack((prediction_arrays)))
    if model_names: predictionsDf.columns = ['ground_truth'] + model_names
    else: predictionsDf.columns = ['ground_truth'] + predictionsDf.columns[1:]
    return calc_metrics(predictionsDf, **calc_metrics_kwargs)