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


import torch
import torch.nn as nn
import optuna
import torch.optim as optim
from Models.GRU import GRUModel  # Assuming GRUModel is defined in Models/GRU.py

def optuna_tune_and_train(
    model_class,  # Model class (e.g., GRUModel)
    train_loader,
    val_loader,
    device,
    input_size,  # The number of features in the dataset
    max_epochs=50,  # Maximum number of epochs for training
    model_save_path="models",  # Path to save the trained models
    model_name="Model",  # The name of the model for saving
    use_best_hyperparams=False,  # Whether to use the best hyperparameters found
    n_trials=20,  # Number of trials for Optuna
    verbose=False  # Whether to print progress during training
):
    """
    Performs hyperparameter optimization using Optuna, trains the model with the best parameters, 
    and saves the best model to the specified path.
    """

    # Optuna's objective function to optimize hyperparameters
    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", 64, 256)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        # Initialize the model with the trial's hyperparameters
        model = model_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=1
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        # Train and validate the model for a few epochs
        for epoch in range(max_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate_logits(model, val_loader, criterion, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f" Early stopping at epoch {epoch}")
                    break

            if verbose:
                print(f"Epoch {epoch + 1}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        return best_val_loss

    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Get the best hyperparameters from the study
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # Build the final model with the best hyperparameters
    best_model = model_class(
        input_size=input_size,
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        dropout=best_params.get("dropout", 0.0),
        output_size=1
    ).to(device)

    optimizer = optim.Adam(best_model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    criterion = nn.MSELoss()

    # Final training loop with the best hyperparameters
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    for epoch in range(max_epochs):
        train_loss = train_epoch(best_model, train_loader, criterion, optimizer, device)
        val_loss = validate_logits(best_model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(best_model.state_dict(), os.path.join(model_save_path, f"{model_name}_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" Early stopping at epoch {epoch}")
                break

        if verbose:
            print(f"Epoch {epoch + 1}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Load the best model (the one with the lowest validation loss)
    best_model.load_state_dict(torch.load(os.path.join(model_save_path, f"{model_name}_best.pt")))

    return best_model, best_params