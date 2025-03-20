import time
import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from EarlyStopping import EarlyStopping
from hyperparameters import OPTUNA_SEARCH_SPACE


def train_epoch(
    model: torch.nn.Module,
    dataLoader: torch.utils.data.DataLoader,
    lossFn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batchStatusUpdate: int = None,
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
        train_loss = 0

        #TODO: Have this use train_epoch
        for batch, batch_data in enumerate(dataLoaderTrain):
            # Unpacks batch data into inputs and targets; can handle arbitrary numbers of inputs
            *inputs, targets = batch_data
            inputs = [input.to(device) for input in inputs]
            targets = targets.to(device)
            y_pred = model(*inputs)

            optimizer.zero_grad()
            loss = lossFn(y_pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

            #inform user of the current batch number when requested
            if batchStatusUpdate and batch % batchStatusUpdate == 0:
                print(f"\tBatch: {batch}")

        train_loss /= len(dataLoaderTrain)
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

def optuna_tune_and_train(
    model_class,  # Any model class (GRU, LSTM, Transformer, etc.)
    train_loader,
    val_loader,
    device,
    max_epochs=50,
    model_save_path="models",
    model_name="Model",
    use_best_hyperparams=False,  # Set False to force a fresh Optuna run
    n_trials=20,  # Number of Optuna trials
    verbose=False  # Whether or not to print out progress
):
    """
    Runs Optuna hyperparameter tuning and trains the model with the best parameters.
    """

    # Step 1: Run Optuna Hyperparameter Tuning
    study = optuna.create_study(direction="minimize", study_name=f"{model_name if model_name != 'Model' else model_class.__name__}_hyperparameter_optimisation")

    def objective(trial):
        """Objective function for Optuna hyperparameter tuning."""
        hidden_size = trial.suggest_int("hidden_size", *OPTUNA_SEARCH_SPACE["hidden_size"])
        num_layers = trial.suggest_int("num_layers", *OPTUNA_SEARCH_SPACE["num_layers"])
        learning_rate = trial.suggest_float("lr", *OPTUNA_SEARCH_SPACE["lr"], log=True)

        # Initialize model
        model = model_class(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Train for a few epochs to evaluate performance
        num_epochs = 10  # Shorter tuning period
        total_loss = 0
        model.train()
        
        for epoch in range(num_epochs):
            # Use extant function to train for one epoch, reporting back the average loss on each item
            avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)

            trial.report(avg_loss, epoch)

            # Handle pruning (early stopping within Optuna)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return avg_loss

    # Run Optuna trials
    if verbose: print(" Running Optuna hyperparameter tuning...")
    study.optimize(objective, n_trials=n_trials)

    # Get Best Hyperparameters
    best_params = study.best_params
    if verbose: print(f" Best hyperparameters found: {best_params}")

    # Step 2: Train Model with Best Hyperparameters
    model = model_class(
        input_size=1,
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        output_size=1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])
    lossFn = nn.MSELoss()

    metadata = train_model(
        model=model,
        maxEpochs=max_epochs,
        modelSavePath=model_save_path,
        modelName=model_name,
        dataLoaderTrain=train_loader,
        dataLoaderValid=val_loader,
        lossFn=lossFn,
        optimizer=optimizer,
        device=device,
        verbose=True
    )

    # Save Final Model
    torch.save(model.state_dict(), os.path.join(model_save_path, f'{model_name}.pth'))
    if verbose: print(" Model training complete and saved!")

    return model, metadata