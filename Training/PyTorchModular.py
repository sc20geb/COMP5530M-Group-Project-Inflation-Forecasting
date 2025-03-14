import torch
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from EarlyStopping import EarlyStopping
import time
import pickle
import os


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
    for batch, (X, Y) in enumerate(dataLoader):

        # Put the data on the appropiate device:
        X = X.to(device)
        Y = Y.to(device)

        # Forward pass:
        y_pred = model(X)

        # Calculate the loss:
        loss = lossFn(y_pred, Y)

        # Update training loss:
        trainLoss += loss.item() * Y.shape[0]
        nSamples += Y.shape[0]

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
        for batch, (X, Y) in enumerate(dataLoader):

            # Put the data on the appropiate device:
            X = X.to(device)
            Y = Y.to(device)

            # Get the predicted Logits
            logits = model(X)

            # calculate the loss:
            loss = lossFn(logits, Y)
            validLoss += loss.item() * Y.shape[0]
            nSamples += Y.shape[0]
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
    }

    stopper = EarlyStopping()
    best_val_loss = float("inf")

    torch.manual_seed(seed)

    for epoch in tqdm(range(0, maxEpochs), desc="Training Progress"):

        t0 = time.time()
        model.train()
        train_loss = 0

        for batch, batch_data in enumerate(dataLoaderTrain):
            if len(batch_data) == 3:  # Case where exogenous variables exist
                X, X_exog, Y = batch_data
                X, X_exog, Y = X.to(device), X_exog.to(device), Y.to(device)
                y_pred = model(X, X_exog)
            else:  # Case where only X, Y exist (no exogenous variables)
                X, Y = batch_data
                X, Y = X.to(device), Y.to(device)
                y_pred = model(X)

            optimizer.zero_grad()
            loss = lossFn(y_pred, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

            #inform user of the current batch number when requested
            if batchStatusUpdate:
                if batch % batchStatusUpdate == 0:
                    print(f"\tBatch: {batch}")

        train_loss /= len(dataLoaderTrain)
        metaData["trainLoss"].append(train_loss)

        model.eval()
        valid_loss = 0
        with torch.inference_mode():
            for batch in dataLoaderValid:
                if len(batch) == 3:  # Exogenous variables present
                    X, X_exog, Y = batch
                    X, X_exog, Y = X.to(device), X_exog.to(device), Y.to(device)
                    y_pred = model(X, X_exog)
                else:  # No exogenous variables
                    X, Y = batch
                    X, Y = X.to(device), Y.to(device)
                    y_pred = model(X)

                loss = lossFn(y_pred, Y)
                valid_loss += loss.item() * Y.shape[0]

        valid_loss /= len(dataLoaderValid)
        # Ensure 'data' is a dictionary
        if not isinstance(metaData, dict): 
            raise TypeError("Expected 'metaData' to be a dictionary, but got: ", type(metaData)) 
        
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