import torch
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from EarlyStopping import EarlyStopping
import time
import pickle


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
    verbose=None,
    seed=42,
):
    """
    This function trains a model for given number of epochs.

    Parameters
    ----------
    model: The model that needs to be trained

    maxEpochs: The maximum amount of epochs to train the model (assuming early stopping does not occur).

    modelSavePath: The path to the directory where the model gets saved for each epoch

    modelName: The name of the model (Used for saving the model).

    dataLoaderTrain: The dataloader used to train the model

    dataLoaderValid: The dataloader of the validation data to evaluate the model (NOT test data).

    lossFn: Loss function to evalute the model

    optimizer: The optimizer used to improve the model

    device: The device to run the model on

    scheduler (optional): The learning rate scheduler to be used, if any

    seed (optional): Sets the random state of the model for reproducibility. Defaults to 42.
    NOTE: random state may not be excactly the same as CUDA has its own randomness on the graphics card.

    Returns:
    --------
    Returns dictionary of traing Data
    """

    data = {
        "trainLoss": [],
        "validLoss": [],
        "times": [],
    }  # used to store data about training for all epochs

    stopper = EarlyStopping()  # initializes early stopping

    torch.manual_seed(seed)  # sets random state

    # loop over each epoch
    for epoch in tqdm(range(0, maxEpochs)):

        t0 = time.time()  # sets timer for epoch

        # Train model for one epoch
        trainLoss = train_epoch(
            model, dataLoaderTrain, lossFn, optimizer, device, batchStatusUpdate
        )

        data["times"].append(time.time() - t0)  # add training time for epoch

        data["trainLoss"].append(trainLoss)  # add average trainloss for epoch

        # validate model:
        validLoss = validate_logits(model, dataLoaderValid, lossFn, device)

        data["validLoss"].append(validLoss)  # add validation Loss for epoch

        # Inform user of training loss and validation loss:
        if verbose is not None:
            print(f"Train Loss epoch {epoch}: {trainLoss}")
            print(f"Valid Loss epoch {epoch}: {validLoss}")

        # Pass validation loss to scheduler, if provided
        if scheduler: scheduler.step(validLoss)

        # Check if model needs to stop early:
        if stopper(model, validLoss):
            # resore the weights with the best validation loss:
            stopper.restoreBestWeights(model)

            # Save traing data:
            with open(
                f"{modelSavePath}/{modelName}_data_stopped.pickle", "wb"
            ) as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Save model weights: NOTE: informs the user which epoch had the best validation loss by adding it to the name of the weights file
            torch.save(
                model.state_dict(),
                f"{modelSavePath}/{modelName}_BEST_STOPPED_AT_{epoch-stopper.counter}.pth",
            )
            print(
                f"Stopped at epoch: {epoch}\nBest weights at epoch: {epoch-stopper.counter}"
            )
            # return data for training
            return data

        # save model at each epoch (incase of failure):
        torch.save(model.state_dict(), f"{modelSavePath}/{modelName}_latest.pth")

    # If model has reached maxEpochs, return data and save model NOTE: model name will have "_UNSTOPPED" appended to inform model has not converged.
    torch.save(model.state_dict(), f"{modelSavePath}/{modelName}_UNSTOPPED.pth")

    # Save traing data:
    with open(f"{modelSavePath}/{modelName}_data_unstopped.pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return data for training
    return data


def loss_curve(trainLoss: list, validLoss: list, title: str = None):
    """
    This function graphs the training loss and validation loss over epochs

    Parameters:
    -----------
    trainLoss: List of training loss's in ascending order of epochs

    validLoss: List of validation loss's in ascending order of epochs

    title: title of graph. If None, then no title is given

    Returns:
    --------
    Void
    """
    epochs = list(range(1, len(trainLoss) + 1))
    plt.plot(epochs, trainLoss, label="train loss")
    plt.plot(epochs, validLoss, label="valid loss")

    plt.xlabel("Epochs")
    plt.ylabel("loss")

    if title is not None:
        plt.title(title)

    plt.legend()
    plt.show()
