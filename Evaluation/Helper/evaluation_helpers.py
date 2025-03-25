import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

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
        predsDf[get_model_name(i.name)]= np.load(i)[:48]
    
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
    predictionsDf: a pandas datframe containg the ground truth and all the predictions for each model, organized in columns.

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
    for model in predictionsDf.columns.drop('ground_truth'):#
        # Calculate the metrics and add them to the metrics dataframe
        for metric in metrics:
            metricsDf.loc[model, metric] = metrics[metric](predictionsDf['ground_truth'].iloc[:horizon], predictionsDf[model].iloc[:horizon])

    return metricsDf