import time
import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from Training.Helper.EarlyStopping import EarlyStopping
from Training.Helper.hyperparameters import OPTUNA_SEARCH_SPACE


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

def optuna_trial_get_kwargs(trial, search_space):
    '''
    Returns suggested variables of the specified type as a kwarg dictionary.
    
    Parameters:
    -----------
    trial: Optuna trial with which to suggest the variables requested.
    search_space: Dictionary with entries of the form {keyword: (type, (lowerbound, upperbound))}.

    Returns:
    --------
    kwargs: Dictionary of keyword arguments containing the values within the ranges provided suggested by the passed optuna trial.
    '''
    kwargs = {}
    for key in search_space:
        type, range = search_space[key]
        if type in [int, 'int']: kwargs[key] = trial.suggest_int(key, *range)
        elif type in [float, 'float']: kwargs[key] = trial.suggest_float(key, *range)
        elif type in [str, 'categorical']: kwargs[key] = trial.suggest_categorical(key, range)
        elif type == 'discrete_uniform': kwargs[key] = trial.suggest_discrete_uniform(key, *range)
        elif type == 'uniform': kwargs[key] = trial.suggest_uniform(key, *range)
        elif type == 'loguniform': kwargs[key] = trial.suggest_loguniform(key, *range)
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

def optuna_tune_and_train(
    model_class,
    train_loader,
    val_loader,
    device,
    model_search_space,
    model_invariates,
    optim_search_space,
    max_epochs=50,
    model_save_path='.',
    model_name="",
    has_optimiser=True,
    n_trials=20,
    n_epochs_per_trial=10,
    return_study=False,
    verbose=False
):
    """
    Runs Optuna hyperparameter tuning and trains the model with the best parameters.

    Parameters:
    -----------
    model_class: The type of model to have hyperparameters optimised (e.g. GRU, LSTM, Transformer, etc.)
    train_loader: DataLoader for training data
    val_loader: DataLoader for validation data (provides validation loss that will be optimised)
    device: torch Device on which model will be optimised
    model_search_space: Dictionary, search space (keyword arguments mapped to a tuple containing type and range of argument) for the model
        e.g. {"hidden_size": (int, (32, 256)), "num_layers": (int, (1, 4))}
    model_invariates: Dictionary, model keyword arguments that do not change
    optim_search_space: Dictionary, search space for the optimiser (same format as model_search_space)
    max_epochs: Optional integer, the maximum number of epochs the model will train with best found hyperparmeters
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

    # Step 1: Run Optuna Hyperparameter Tuning
    study = optuna.create_study(direction="minimize",
                                # Uses successive halving several times at different levels of pruning aggressiveness depending on whether validation performance of configurations are distinguishable after a given number of runs (per-config resource allocation)
                                pruner=optuna.pruners.HyperbandPruner(),
                                study_name=f"{model_name if model_name else model_class.__name__}_hyperparameter_optimisation")

    def objective(trial):
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

    # Run Optuna trials
    if verbose: print("Running Optuna hyperparameter tuning...")
    study.optimize(objective, n_trials=n_trials)

    # Get the best hyperparameters from the study
    best_params = study.best_params
    if verbose: print(f"Best hyperparameters found: {best_params}")

    # Split the best parameters found by the trial into the model's and the optimiser's
    if has_optimiser: model_best_params, optimiser_best_params = split_params(best_params, model_search_space, optim_search_space)
    else: model_best_params = best_params

    # Build the final model with the best hyperparameters
    #TODO: Ensure has_optimiser stuff works - i.e. compatability with DARTS models
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