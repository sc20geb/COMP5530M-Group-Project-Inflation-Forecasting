import time
import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import pytorch_lightning
from darts.utils.likelihood_models import QuantileRegression
from darts.metrics import mse
import pandas as pd
import json

import sys
sys.path.append(os.path.join('..', '..'))  #ensures imports can be performed from anywhere in the repository

from Training.Helper.EarlyStopping import EarlyStopping  #added prefix so import was recognised

# Max depth of recursive calls that should be restricted below the default recursion limit
MAX_DEPTH = 10
# Horizons for which all models will be optimised
HORIZONS = [1, 3, 6, 12]


def train_epoch(
    model: torch.nn.Module,
    dataLoader: torch.utils.data.DataLoader,
    lossFn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batchStatusUpdate: int = None,
    gradientClipper: torch.nn.utils = None,
    gradientClipperKwargs: dict = {},
):
    """
    This function trains a model for one epoch

    Parameters
    ----------
    model: The intitialized model that needs to be trained

    dataLoader: the initialized dataloader used to train the model

    lossFn: Initialized Loss function to calculate the loss.

    optimizer: The initialized optimizer used to improve the model

    device: The device to run the model on .i.e "cuda" or "cpu"

    batchStatusUpdate: Informs the user what batch they are on every multiple of batchStatusUpdate. Use None to disable.

    gradientClipper: (optional) a PyTorch gradient clipper

    gradientClipperKwargs: (optional) Dictionary specifying any keyword arguments to gradientClipper

    Returns:
    --------
    average training Loss of epoch
    """

    # Enter training mode:
    model.train()

    # initialize training loss to zero and nSamples to zero at start of each epoch
    trainLoss = 0  # accumulates the total loss for epoch
    nSamples = 0  # accumulates number of samples per epoch (divides the total loss to find average loss)

    # loop over each batch:
    for batch, batch_data in enumerate(dataLoader):

        # Unpacks batch data into inputs and targets; can handle arbitrary numbers of inputs
        *inputs, targets = batch_data

        # Put the data on the appropriate device
        inputs = [input.to(device) for input in inputs]
        targets = targets.to(device)

        # Forward pass:
        y_pred = model(*inputs)

        # Calculate the loss:
        loss = lossFn(y_pred, targets)

        # Update training loss:
        trainLoss += loss.item() * targets.shape[0]
        nSamples += targets.shape[0]

        # Remove previous gradients:
        optimizer.zero_grad()

        # Backwards pass:
        loss.backward()

        # Apply torch.nn utilities gradient clipper function if passed:
        if gradientClipper: gradientClipper(model.parameters(), **gradientClipperKwargs)

        # Improve model:
        optimizer.step()

        # Inform user of which  batch it is on
        if batchStatusUpdate is None:
            pass
        elif batch % batchStatusUpdate == 0:
            print(f"\tBatch: {batch}")

    return trainLoss / nSamples


def validate_logits(
    model: torch.nn.Module,
    dataLoader: torch.utils.data.DataLoader,
    lossFn: torch.nn.Module,
    device: torch.device,
):
    """
    This function validates the model used in each epoch.
    NOTE: assumes the output of the model is logits i.e. output before
    last activation function (softmax for multi-class classification, sigmoid
    for binary/multi-label classification)

    Parameters
    ----------
    model: the model to be evaluated.

    dataLoader: The dataLoader to evaluate the model.

    LossFn: used to calculate the error on the dataset.

    Returns:
    --------
    average validation Loss of epoch

    """
    # initialize validation loss and number of samples as zero. (Used for calculating the average validation loss)
    validLoss = 0
    nSamples = 0

    # Put the model in evaluation mode:
    model.eval()

    # Loop over each batch in the validation set.
    with torch.inference_mode():
        for batch, batch_data in enumerate(dataLoader):
            # Unpacks batch data into inputs and targets; can handle arbitrary numbers of inputs
            *inputs, targets = batch_data

            # Put the data on the appropriate device
            inputs = [input.to(device) for input in inputs]
            targets = targets.to(device)

            # Get the predicted Logits
            logits = model(*inputs)

            # calculate the loss:
            loss = lossFn(logits, targets)
            validLoss += loss.item() * targets.shape[0]
            nSamples += targets.shape[0]
            # TODO: Implement torch metrics feature:

    return validLoss / nSamples


def train_model(
    model: torch.nn.Module,
    maxEpochs: int,
    modelSavePath: str,
    modelName: str,
    dataLoaderTrain: torch.utils.data.DataLoader,
    dataLoaderValid: torch.utils.data.DataLoader,
    lossFn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
    batchStatusUpdate=None,
    verbose=True,
    seed=42,
    gradientClipper=torch.nn.utils.clip_grad_norm_,
    gradientClipperKwargs={'max_norm': 1.0},
):
    """
    This function trains a model for given number of epochs and saves the best performing model.

    Parameters:
    ----------
    model: The model that needs to be trained.
    maxEpochs: Maximum number of epochs to train the model.
    modelSavePath: Directory where the best model will be saved.
    modelName: Name of the model for saving.
    dataLoaderTrain: DataLoader for training data.
    dataLoaderValid: DataLoader for validation data.
    lossFn: Loss function to evaluate the model.
    optimizer: Optimizer to improve the model.
    device: Device to run the model on ("cuda" or "cpu").
    scheduler: (Optional) Learning rate scheduler.
    batchStatusUpdate: (Optional) Prints batch status update frequency.
    verbose: (Optional) Whether to print loss updates.
    seed: (Optional) Random seed for reproducibility.
    gradientClipper: (Optional) a PyTorch gradient clipper.
    gradientClipperKwargs: (Optional) Dictionary specifying any keyword arguments to gradientClipper.

    Returns:
    --------
    Returns dictionary containing:
    - 'trainLoss': List of training losses for each epoch.
    - 'validLoss': List of validation losses for each epoch.
    - 'best_model_path': Path to the saved best model.
    """
    os.makedirs(modelSavePath, exist_ok=True)
    best_model_path = None  # Initialize best model path

    metaData = {
        "trainLoss": [],
        "validLoss": [],
        "times": [],
        "best_model_path": "",
    }

    stopper = EarlyStopping()
    best_val_loss = float("inf")

    torch.manual_seed(seed)

    for epoch in tqdm(range(0, maxEpochs), desc="Training Progress"):
        t0 = time.time()
        model.train()

        train_loss = train_epoch(model, dataLoaderTrain, lossFn, optimizer, device, batchStatusUpdate=batchStatusUpdate, 
                                 gradientClipper=gradientClipper, gradientClipperKwargs=gradientClipperKwargs)

        metaData["trainLoss"].append(train_loss)

        valid_loss = validate_logits(model, dataLoaderValid, lossFn, device)
        metaData["validLoss"].append(valid_loss)
        metaData["times"].append(time.time() - t0)

        if verbose:
            print(f"Epoch {epoch+1}/{maxEpochs} - Train Loss: {train_loss:.6f}, Val Loss: {valid_loss:.6f}")

        if scheduler:
            scheduler.step(valid_loss)

        # Save best model dynamically
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model_path = os.path.join(modelSavePath, f"{modelName}_BEST_STOPPED_AT_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            metaData["best_model_path"] = best_model_path
            print(f"Best model saved at {best_model_path} (Epoch {epoch+1})")

        if stopper(model, valid_loss):
            stopper.restoreBestWeights(model)
            print(f"Early stopping at epoch {epoch+1}. Best model restored.")
            return metaData

    print(f"Training completed. Best model saved at {best_model_path}")
    return metaData
    
def loss_curve(trainLoss: list, validLoss: list, title: str = None):
    """
    This function graphs the training loss and validation loss over epochs.

    Parameters:
    -----------
    trainLoss: List of training losses in ascending order of epochs.
    validLoss: List of validation losses in ascending order of epochs.
    title: (Optional) Title of the graph.

    Returns:
    --------
    Void (Displays the graph).
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(trainLoss) + 1), trainLoss, label="Training Loss", color="blue")
    plt.plot(range(1, len(validLoss) + 1), validLoss, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title if title else "Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def split_arg_list(lst : list):
    '''
    Splits a list of arguments into a list of regular arguments and a single keyword argument dictionary. (both are blank if not found)
        Note: If several dictionaries are included in the list, the final dictionary is considered the definitive dictionary of keyword arguments.
    
    Parameters:
    -----------
    lst: list containing the arguments to be split.

    Returns:
    --------
    (list of arguments, dictionary of keyword arguments).
    '''
    args = []
    kwargs = {}
    for el in lst:
        if type(el) == dict: kwargs = el
        else: args.append(el)
    return args, kwargs

def optuna_trial_get_kwargs(trial, search_space, cur_depth=0):
    '''
    Returns suggested variables of the specified type as a kwarg dictionary.
    Search spaces should be formatted as below:
    {'example_hyperparameter_name': (type, *args, kwargs), ...}
        where type is in [int, float, str, 'discrete_uniform', 'uniform', 'loguniform'],
        args is an arbitrary number of arguments to the suggest_ function (e.g. 1, 10 for a lower and upper bound), and
        kwargs is a dictionary of keyword arguments to the suggest_ function containing: {keyword: argument, ...}
            where keywords are strings and arguments are their associated values
    
    Parameters:
    -----------
    trial: Optuna trial with which to suggest the variables requested.
    search_space: Dictionary with entries of the form {keyword: (type, lowerbound, upperbound, ...)}.

    Returns:
    --------
    kwargs: Dictionary of keyword arguments containing the values within the ranges provided suggested by the passed optuna trial.
    '''
    if cur_depth > MAX_DEPTH: raise RecursionError(f'Cannot exceed recursion depth of {MAX_DEPTH}')
    kwargs = {}
    for key in search_space:
        # If the type, arguments, and keyword arguments are found immediately, suggest values for them and fill in dictionary
        # Otherwise, recursively call this function to find the type and range of sub-dictionaries (until the macro-defined depth)
        if type(search_space[key]) == dict:
            kwargs[key] = optuna_trial_get_kwargs(trial, search_space[key], cur_depth=cur_depth+1)
            continue
        else:
            args_plus_type, func_kwargs = split_arg_list(search_space[key])
            ty, *args = args_plus_type
        # Ask Optuna to suggest a value for each type and arguments found
        if ty in [int, 'int']: kwargs[key] = trial.suggest_int(key, *args, **func_kwargs)
        elif ty in [float, 'float']: kwargs[key] = trial.suggest_float(key, *args, **func_kwargs)
        elif ty in [str, 'categorical']: kwargs[key] = trial.suggest_categorical(key, *args, **func_kwargs)
        elif ty == 'discrete_uniform': kwargs[key] = trial.suggest_discrete_uniform(key, *args, **func_kwargs)
        elif ty == 'uniform': kwargs[key] = trial.suggest_uniform(key, *args, **func_kwargs)
        elif ty == 'loguniform': kwargs[key] = trial.suggest_loguniform(key, *args, **func_kwargs)
        else:
            message = 'The first element in each value pair in the search space dictionary provided should be either:\n'
            message += 'A type in [int, float, str], or\nA string in [\'int\', \'float\', \'categorical\', \'discrete_uniform\', \'uniform\', \'log_uniform\']'
            raise ValueError(message)
    return kwargs

def split_params(params : dict, search_space1 : dict, search_space2 : dict):
    '''
    Splits the dictionary of parameters provided into two that share the keys of the first and second search spaces provided
    
    Parameters:
    -----------
    params: Dictionary containing the parameters to be split.
    search_space1: Dictionary containing the keys included in the first split.
    search_space2: Dictionary containing the keys included in the second split.

    Returns:
    --------
    (Dictionary containing first split, Dictionary containing second split).
    '''
    return {key: params[key] for key in params if key in search_space1}, {key: params[key] for key in params if key in search_space2}

def optuna_tune_and_train_pytorch(
    model_class,
    train_loader,
    val_loader,
    device,
    model_search_space,
    model_invariates,
    optim_search_space,
    max_epochs=50,
    objective=None,
    model_save_path='.',
    model_name="",
    has_optimiser=True,
    n_trials=20,
    n_epochs_per_trial=10,
    return_study=False,
    verbose=False
):
    """
    Runs Optuna hyperparameter tuning for PyTorch models and trains the model with the best parameters.

    Parameters:
    -----------
    model_class: The type of model to have hyperparameters optimised (e.g. GRU, LSTM, Transformer, etc.)
    train_loader: DataLoader for training data
    val_loader: DataLoader for validation data (provides validation loss that will be optimised)
    device: torch Device on which model will be optimised
    model_search_space: Dictionary, search space (keyword arguments mapped to a tuple containing type and range of argument) for the model
        e.g. {"hidden_size": (int, 32, 256), "num_layers": (int, 1, 4)}
    model_invariates: Dictionary, model keyword arguments that do not change
    optim_search_space: Dictionary, search space for the optimiser (same format as model_search_space)
    max_epochs: Optional integer, the maximum number of epochs the model will train with best found hyperparmeters
    objective: Optional objective function to be used in the Optuna study instead of the default
    model_save_path: Optional string, the file path to which the results of the final model's training will be saved
    model_name: Optional string, the name of the model used to name the Optuna study
    has_optimiser: Optional boolean, defines whether the model being trained should be trained with an optimiser
    n_trials: Optional integer, the number of Optuna trials to perform
    n_epochs_per_trial: Optional integer, number of epochs trained per Optuna trial
    return_study: Optional boolean, whether to return the Optuna study performed
    verbose: Optional boolean, whether or not to print out progress

    Returns:
    --------
    Best-parameter trained model, training metadata, (optional) Optuna study performed.
    """
    def obj(trial):
        """Objective function for Optuna hyperparameter tuning."""

        model_kwargs = optuna_trial_get_kwargs(trial, search_space=model_search_space)

        # Initialize model, optimiser, and criterion
        model = model_class(**model_invariates, **model_kwargs).to(device)
        if has_optimiser:
            optim_kwargs = optuna_trial_get_kwargs(trial, search_space=optim_search_space)
            optimizer = optim.Adam(model.parameters(), **optim_kwargs)
        criterion = nn.MSELoss()

        # Train for a few epochs to evaluate performance
        model.train()
        
        for epoch in range(n_epochs_per_trial):
            # Use extant function to train for one epoch, reporting back the average loss on each item
            avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)

            # Decide hyperparameters on validation loss minimisation, not training loss minimisation
            valid_loss = validate_logits(model, val_loader, criterion, device)

            trial.report(valid_loss, epoch)

            # Handle pruning (early stopping within Optuna)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return valid_loss
    
    if return_study: best_params, study = optuna_tune(obj, model_name, n_trials, verbose, return_study=return_study)
    else: best_params = optuna_tune(obj, model_name, n_trials, verbose, return_study=return_study)

    # Split the best parameters found by the trial into the model's and the optimiser's
    if has_optimiser: model_best_params, optimiser_best_params = split_params(best_params, model_search_space, optim_search_space)
    else: model_best_params = best_params

    # Build the final model with the best hyperparameters
    best_model = model_class(**model_invariates, **model_best_params).to(device)

    if has_optimiser: optimizer = optim.Adam(best_model.parameters(), **optimiser_best_params)
    criterion = nn.MSELoss()

    # Final training loop with the best hyperparameters
    metadata = train_model(
        model=best_model,
        maxEpochs=max_epochs,
        modelSavePath=model_save_path,
        modelName=model_name,
        dataLoaderTrain=train_loader,
        dataLoaderValid=val_loader,
        lossFn=criterion,
        optimizer=optimizer,
        device=device,
        verbose=verbose
    )

    # Save Final Model
    torch.save(best_model.state_dict(), os.path.join(model_save_path, f'{model_name}_best.pth'))
    if verbose: print("Model training complete and saved!")

    if return_study: return best_model, metadata, study
    return best_model, metadata


def load_prediction(input_size:int, df_exog:pd.DataFrame,target_series:pd.Series,exog_scaler:Scaler, target_scaler:Scaler, pred_date:pd.DatetimeIndex):
    '''
    This function loads an input_size amount of data points before pred_date from df_exog and target_series and instantiates a darts TimeSeries object with scaled values
    from the relevant fitted scalers.
    NOTE: target_series and df_exog MUST have a date-time index.

    Parameters:
    -----------
    input_size: An integer representing the number of previous data points to be loaded.

    df_exog: A pandas dataframe with a date-time index, representing a dataframe of exogenous variables.

    target_series: A pandas Series, representing the target time series.

    exog_scaler: darts Scaler Object for the exogenous variables

    target_scaler: darts Scaler Object for the target series 

    Pred_date: The date which needs to be predicted. 

    Returns:
    --------
    Returns Darts TimeSeries object of Scaled exogenous variables and scaled target time series of length: input_size, before pred_date.
    '''
    X=TimeSeries.from_dataframe(df_exog.loc[pred_date-pd.DateOffset(months=input_size):pred_date-pd.DateOffset(months=1),:], freq=pd.DateOffset(months=1))
    y_past= TimeSeries.from_series(target_series.loc[pred_date-pd.DateOffset(months=input_size):pred_date-pd.DateOffset(months=1)], freq=pd.DateOffset(months=1))

    return exog_scaler.transform(X), target_scaler.transform(y_past)

def train_valid_split_darts(exog_df:pd.DataFrame, target_series:pd.Series, valid_size:int,input_size:int):
    '''
    Splits the data into a training and validation set, where the validation has a partial overlap with the training data (NOT the other way around) to minimize data wastage.
    train set size= total size - valid_size
    valid set size = valid_size + input_size

    Parameters:
    -----------
    exog_df: Pandas Dataframe containing the exogenous variables.

    target_series: Pandas Series containing the target time series.

    valid_size: The number of predictions the validation set should have.

    input_size: The number of previous time points required to make a prediction.

    Returns:
    --------
    Returns train_target, valid_target, train_exog, valid_exog
    '''

    train_target= TimeSeries.from_series(target_series.iloc[:-valid_size])
    valid_target= TimeSeries.from_series(target_series.iloc[-valid_size-input_size:])

    train_exog= TimeSeries.from_dataframe(exog_df.iloc[:-valid_size,:])
    valid_exog= TimeSeries.from_dataframe(exog_df.iloc[-valid_size-input_size:,:])

    return train_target, valid_target, train_exog, valid_exog

def optuna_tune(objective, model_name : str, n_trials : int, verbose : bool, return_study : bool = False,
                existing_trials : list[optuna.Trial] = [], existing_params : list[dict] = []):
    '''
    General Optuna objective function optimisation framework.

    Parameters:
    -----------
    objective: Optuna objective function to be optimised (takes a single trial as argument).
    model_name: The name of the model being optimised; used to name the optuna study.
    n_trials: The number of Optuna trials to perform.
    verbose: Whether to display intermediate progress of trials/ final best hyperparameters.
    return_study: Whether or not to return the Optuna study performed.
    existing_trials: Optional list of trials previously calculated with which to initialise optuna.
    existing_params: Optional list of dictionaries representing existing parameters discovered by previous optuna run(s) with which optuna will be initialised.

    Returns:
    --------
    Dictionary of optimal parameters found, (optional) Optuna study performed
    '''
    
    # Define optuna study and optimize
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.HyperbandPruner(), study_name=f"{model_name}_hyperparameter_optimisation")
    if existing_trials: study.add_trials(existing_trials)
    if existing_params:
        for param in existing_params: study.enqueue_trial(param)
    
    # Optimise the objective provided using the study created
    # NOTE: Assumes that the arguments for 'objective' are already defined in the calling function
    if verbose: print("Running Optuna hyperparameter tuning...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    # Get and return the best parameters
    best_params = study.best_params
    if verbose: print(f"Best hyperparameters found:\n{best_params}")
    if return_study: return best_params, study
    return best_params

def fix_model_kwargs(model_kwargs, model_name, n_epochs, early_stopper):
    model_kwargs["n_epochs"]=n_epochs
    model_kwargs["model_name"]=model_name
    # Darts allows probabilistic/deterministic variants (likelihood=(...),loss_fn= None for probabilistic)
    if 'loss_fn' in model_kwargs.keys():
        if model_kwargs['loss_fn']== 'QuantileRegression':
            model_kwargs['loss_fn']= None
            model_kwargs['likelihood']=QuantileRegression((0.25,0.5,0.75))
        else:
            model_kwargs['likelihood']=None
            model_kwargs['loss_fn']= nn.MSELoss()
    # Add the most generic trainer kwargs that work for everyone, including the passed early stopper
    model_kwargs["pl_trainer_kwargs"]={
            "accelerator":'auto',
            "callbacks": [early_stopper]}
    
    return model_kwargs

def darts_optuna(model_cls:object,model_name:str,model_search_space:dict,
                  invariates_kwargs:dict, target_series:pd.Series,
                  exog_df:pd.DataFrame,valid_size:int,horizon:int,
                  n_trials:int=100, n_epochs:int=10000, patience:int=5, tol:float=1e-5,
                  existing_trials:list[optuna.Trial]=[], existing_params:list[dict]=[],
                  verbose:bool=True):
    '''
    Finds optimal hyper-parameters for a given model class (model_cls) and a search space (model_search_space).
    NOTE: target_series and exog_df should include all availible data and should therefore not be split into a validation set (test set should still be separate).
    NOTE: Valid size is expected to be an int, representing the number of validation predictions. Therefore valid_size>=horizon.

    Parameters:
    -----------
    model_cls: the class fingerprint of the darts model.

    model_name: Name of the model (used for saving the best weights for early stopping)

    model_search_space: a dictionary, where the key is a string of a models key word argument, which needs to be optimized. The values are the Optuna reccomendations.

    invariates_kwargs: a dictionary containing other model parameters which do NOT need to be optimized.

    target_series: Pandas series containing the target Time series NOTE: this includes the validation data and should therefore not be splitted.

    exog_df: Pandas Dataframe containing the exogenous variables.

    valid_size: an int representing the number of validation predictions to use. NOTE: Valid size is expected to be an int, representing the number of validation predictions.
    Therefore valid_size>=horizon. It is also worth noting that it takes into account the input size of the model. Therfore there will be a partial overlap with the test set
    with inputs, however the outputs are completely separate. This allows for minimal data wastage and still prervents data leakage.

    horizon: the number of time steps into the future the model should optimize for.

    n_trials: Number of optuna trials to perform (greater the number the better the results, but at the cost of time).

    n_epochs: Maximum number of epochs the model should fit for (usually early stopping stops before this).

    patience: The number of epochs to wait with no improvement of tol, before early stopping.

    tol: the minimum increase in performance which each epoch requires otherwise early stopping occurs (see also patience).

    existing_trials: Optional list of trials previously calculated with which to initialise optuna.
    
    existing_params: Optional list of dictionaries representing existing parameters discovered by previous optuna run(s) with which optuna will be initialised.

    verbose: Used to print the final results.

    Returns:
    --------
    The optimized hyper parameters as a dict.
    NOTE: this does not include the invariates_kwargs.
    '''
    
    if valid_size<horizon:
        raise Exception("valid_size should be at least as big as horizon.")
    
    def objective(trial):
        """Objective function for darts models hyperparameter tuning."""

        # Get the suggested optuna parameters
        model_kwargs = optuna_trial_get_kwargs(trial, search_space=model_search_space)

        # Other compulsory model parameters:
        early_stopper = pytorch_lightning.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=tol,
            mode="min",
            verbose=verbose
        )

        model_kwargs = fix_model_kwargs(model_kwargs, model_name, n_epochs, early_stopper)

        # Define compulsory parameters for optimization
        invariates_kwargs["save_checkpoints"]=True
        invariates_kwargs["force_reset"]=True
        invariates_kwargs["random_state"]=42

        # Split into training and validation set:
        train_target, valid_target, train_exog, valid_exog = train_valid_split_darts(exog_df,target_series,valid_size,model_kwargs["input_chunk_length"])

        # Instantiate scalers and scale
        target_scaler = Scaler()
        exog_scaler = Scaler()

        train_target_scaled = target_scaler.fit_transform(train_target)
        train_exog_scaled = exog_scaler.fit_transform(train_exog)

        valid_target_scaled = target_scaler.transform(valid_target)
        valid_exog_scaled = exog_scaler.transform(valid_exog)        
        
        # Define the model
        model=model_cls(**model_kwargs, **invariates_kwargs)

        #Train the model
        model.fit(
            series=train_target_scaled,
            past_covariates=train_exog_scaled,
            val_series=valid_target_scaled,
            val_past_covariates=valid_exog_scaled,
            verbose=verbose,
        )

        # Load the best weights from early stopping:
        model.load_from_checkpoint(model_name=model_name,best=True)

        # Validate the model:
        rmses=np.array([])# stores the rmse of each validation sample
        # Loop over relevent validation samples (taking into account the horizon): 
        for i in range(target_series.shape[0]-valid_size, target_series.shape[0]-horizon +1):
            #Load the inputs and outputs
            x,y=load_prediction(model_kwargs["input_chunk_length"],exog_df,target_series,exog_scaler,target_scaler,target_series.index[i])
            
            # If the model is probabilistic, take 500 samples and average them
            if model_kwargs['loss_fn']== None:
                pred=target_scaler.inverse_transform(model.predict(n=horizon,series=y,past_covariates=x,verbose=False, num_samples=500)).values().flatten()

            else:
                pred=target_scaler.inverse_transform(model.predict(n=horizon,series=y,past_covariates=x,verbose=False)).values().flatten()
            
            ground_truth=target_series.loc[target_series.index[i]:target_series.index[i+ horizon -1]].to_numpy()

            #claculate rmse for prediction:
            rmses=np.append(rmses,root_mean_squared_error(ground_truth,pred))
        
        #Macro-average rmse of predictions:
        return np.mean(rmses)
    
    return optuna_tune(objective, model_name, n_trials, verbose, existing_trials=existing_trials, existing_params=existing_params)

def save_model_hyper_params(file_name:str, params:dict):
    '''
    This saves the model's best hyper parameters provided.
    (can be used to write any dictionary as a json file)

    '''
    with open(file_name,'w') as f:
        json.dump(params,f)

def load_model_hyper_params(file_name:str):
    '''
    Loads hyperparameters from the file at the path provided.
    (can be used to load any json file)

    '''
    with open(file_name,'r') as f:
        return json.load(f)

def optuna_tune_and_train_darts(model_class,
                                train_target, val_target, train_exo, val_exo,
                                model_search_space, model_invariates,
                                max_epochs=50,
                                objective=None,
                                model_save_path='.', model_name="",
                                n_trials=20, n_epochs_per_trial=10, criterion=mse,
                                return_study=False, verbose=False):
    """
    Runs Optuna hyperparameter tuning for Darts models and trains the model with the best parameters.

    Parameters:
    -----------
    model_class: The type of model to have hyperparameters optimised (e.g. TiDE)
    train_target: TimeSeries with training target data
    val_target: TimeSeries with validation data (provides validation loss that will be optimised)
    model_search_space: Dictionary, search space (keyword arguments mapped to a tuple containing type and range of argument) for the model
        e.g. {'input_chunk_length': (int, 24, 60),'num_encoder_layers': (int, 1, 3),'dropout': (float, 0.1, 0.5, {'log': True}),'optimizer_kwargs': {"lr": (float, 1e-4, 1e-2)}}
    model_invariates: Dictionary, model keyword arguments that do not change
    max_epochs: Optional integer, the maximum number of epochs the model will train with best found hyperparmeters
    objective: Optional objective function to be used in the Optuna study instead of the default
    model_save_path: Optional string, the file path to which the results of the final model's training will be saved
    model_name: Optional string, the name of the model used to name the Optuna study
    n_trials: Optional integer, the number of Optuna trials to perform
    n_epochs_per_trial: Optional integer, number of epochs trained per Optuna trial
    return_study: Optional boolean, whether to return the Optuna study performed
    verbose: Optional boolean, whether or not to print out progress

    Returns:
    --------
    Best-parameter trained model, training metadata, (optional) Optuna study performed.
    """
    
    # Create an optuna study
    study = optuna.create_study(direction="minimize",
                                # Uses successive halving several times at different levels of pruning aggressiveness depending on whether validation performance of configurations are distinguishable after a given number of runs (per-config resource allocation)
                                pruner=optuna.pruners.HyperbandPruner(),
                                study_name=f"{model_name if model_name else model_class.__name__}_hyperparameter_optimisation")

    def obj(trial):
        model_kwargs = optuna_trial_get_kwargs(trial, model_search_space)

        #scaled_train_exo_ranked, scaled_val_exo_ranked = get_optuna_ranked_series(trial, scaled_train_exo_r, scaled_val_exo_r, ranked_features)

        # Initialize the TiDEModel with suggested hyperparameters
        model = model_class(**model_kwargs, **model_invariates)

        # Fit the model
        model.fit(series = train_target,
                past_covariates = train_exo,
                val_series = val_target,
                val_past_covariates = val_exo,
                epochs=n_epochs_per_trial,
                verbose = verbose)

        # Evaluate the model
        # (this is an alternative option for evaluation, where the model must predict the final prediction_size elements of the validation data having been given all other validation data;
        #  if switching to this method, ensure that final prediction is performed with the same setup (this is currently done just by predicting the next n values))
        #scaled_val_predictions = model.predict(n=prediction_size,series=scaled_val_target[:-prediction_size],past_covariates=scaled_val_exo[:-prediction_size], verbose=False)]
        #val_predictions = targetScaler.inverse_transform(scaled_val_predictions, verbose=False)
        #error = mse(val_target[-prediction_size:], val_predictions, verbose=False)

        prediction_size = model.output_chunk_length
        # Raw output is scaled, so inverse transform to become comparable with validation set
        val_predictions = model.predict(n=prediction_size, verbose=verbose)
        # Only uses the first prediction_size values of val_target, since this is the size of the prediction made by the model
        error = criterion(val_target[:prediction_size], val_predictions, verbose=False)
        return error
    
    # Optimise the study
    if verbose: print("Running Optuna hyperparameter tuning...")
    if objective: obj = objective
    study.optimize(obj, n_trials=n_trials)

    best_params = reformat_best_params(study.best_params, model_search_space)
    best_model = model_class(**best_params, **model_invariates)

    best_model.fit(series=train_target,
               past_covariates=train_exo,
               val_series=val_target,
               val_past_covariates=val_exo,
               epochs=max_epochs,
               verbose=verbose)
    
    best_model.save(os.path.join(model_save_path, f"{model_name if model_name else model_class.__name__}.pkl"))
    
    if return_study: return best_model, study
    return best_model

def reformat_best_params(best_params : dict, formatting_dict : dict, cur_depth : int = 0):
    '''
    Reformats the provided dictionary of parameters into the same format as the dictionary provided.
    
    Parameters:
    -----------
    best_params: Dictionary containing the best parameters (extracted from an Optuna study)
    formatting_dict: Dictionary containing the same keys as those of the best parameters, but potentially in a distinct format.

    Returns:
    --------
    Dictionary in the same format as formatting_dict, but containing the values in best_params.
    '''
    if cur_depth > MAX_DEPTH: raise RecursionError(f'Cannot exceed recursion depth of {MAX_DEPTH}')
    reformatted = {}
    for key in formatting_dict:
        if key in best_params: reformatted[key] = best_params[key]
        else: reformatted[key] = reformat_best_params(best_params, formatting_dict[key], cur_depth=cur_depth+1)
    return reformatted
