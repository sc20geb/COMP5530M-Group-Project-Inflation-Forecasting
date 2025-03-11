import torch
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

def minMaxScale(vals : np.array) -> np.array:
    '''
    Performs min-max scaling on the provided numpy array

    Parameters:
    -----------
    vals: numpy array on which to perform scaling

    Returns:
    --------
    min-max scaled numpy array
    '''
    return (vals - np.min(vals)) / (np.max(vals) - np.min(vals))

#TODO: Should full Fourier transformed features (including before and after the sequence) be included for each sequence?
def createSequences(data : np.array, sequence_length : int, other_features=[]) -> tuple[np.array, np.array]:
    '''
    Creates sequences (past sequence_length months → next month)

    Parameters:
    -----------
    data: numpy array used to create the sequence
    sequence_length: length of the sequences requested
    other_features (optional): additional features to include with each X sequence

    Returns:
    --------
    numpy arrays containing the X and y values of each sequence at their corresponding indices
    '''
    X, y = [], []
    for i in range(len(data) - sequence_length):
        if len(other_features) > 0: X.append(np.hstack([data[i : i + sequence_length].flatten(), other_features]))
        else: X.append(data[i : i + sequence_length].flatten())
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def trainValTestSplit(X : np.array, y : np.array, trainSize : float, valSize: float, testSize=None) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    '''
    Splits an input X and y dataset array into training, validation, and test X and y arrays

    Parameters:
    -----------
    X: numpy array containing the input values for a model
    y: numpy array containing the output values for a model
    trainSize: float defining the percentage of the dataset to be assigned to training
    valSize: float defining the percentage of the dataset to be assigned to validation
    testSize (optional): float defining the percentage of the dataset to be assigned to testing (inferred if not given)

    Returns:
    --------
    numpy arrays containing the X and y values of each split of the dataset in the order: training, validation, testing
    '''
    if not testSize: testSize = 1- trainSize+valSize
    if trainSize+valSize+testSize != 1: raise ValueError('Train, validation, and test sizes must add up to 1.')

    train_idx = int(len(X) * trainSize)
    valid_idx = int(len(X) * (trainSize+valSize))

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_valid, y_valid = X[train_idx:valid_idx], y[train_idx:valid_idx]
    X_test, y_test = X[valid_idx:], y[valid_idx:]
    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Load and Format Data for N-BEATS
def load_data(train_file, sequence_length=48):
    df = pd.read_csv(train_file)
    df["observation_date"] = pd.to_datetime(df["observation_date"], format="%m/%Y")
    df = df.sort_values(by="observation_date").reset_index(drop=True)

    target_col = "fred_PCEPI"
    data = df[[target_col]].values.astype(np.float32)

    # Compute Fourier Transform Features for Seasonality
    fft_features = minMaxScale(np.abs(fft(data.flatten()))[:sequence_length])

    X, y = createSequences(data, sequence_length, fft_features)

    # Train-Validation-Test Split (70%-15%-15%)
    X_train, y_train, X_valid, y_valid, X_test, y_test = trainValTestSplit(X, y, 0.7, 0.15, 0.15)

    # Fix Data Leakage: Scale After Splitting
    #TODO: Why does this fix data leakage? Is the leakage from the Fourier transformed features?
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_valid = y_scaler.transform(y_valid.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))

    return X_train, y_train, X_valid, y_valid, X_test, y_test, df["observation_date"][sequence_length:], scaler, y_scaler

# Convert Data to PyTorch Tensors
def prepare_dataloader(X, y, batch_size=32):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)