import torch
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

def apply_fft_per_sequence(data, seq_len, num_components=10):
    '''
    Applies Fast Fourier Transform (FFT) to each sequence in the input time-series data 
    and extracts the real and imaginary components.

    Parameters:
    -----------
    data: numpy array
        The input time-series data, where each row represents a timestamp.
    seq_len: int
        The length of each sequence for which FFT is applied. Determines how many past 
        values are considered in the transformation.

    Returns:
    --------
    fft_real: numpy array
        Log-transformed absolute values of the real component of the FFT.
    fft_imag: numpy array
        Log-transformed absolute values of the imaginary component of the FFT.
    '''
    fft_real, fft_imag = [], []
    for i in range(len(data) - seq_len):
        fft_transformed = fft(data[i : i + seq_len].flatten())
        fft_real.append(np.log1p(np.abs(np.real(fft_transformed[:num_components]))))
        fft_imag.append(np.log1p(np.abs(np.imag(fft_transformed[:num_components]))))
    return np.array(fft_real), np.array(fft_imag)

def add_lagged_features(df, target_col, lags=[1, 3, 6]):
    '''
    Generates lagged features for the given target column.
    
    Lagging means shifting the values of the target column by a certain number of time steps 
    to create past observations as features. This helps the model learn from previous values.

    Parameters:
    -----------
    df: pandas DataFrame
        The input DataFrame containing the target column.
    target_col: str
        The name of the target column for which lag features are created.
    lags: list of int
        A list of lag values indicating how many time steps back to shift the data.

    Returns:
    --------
    df: pandas DataFrame
        The DataFrame with additional lagged columns.
    '''
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df

def add_rolling_features(df, target_col, windows=[3, 6, 12]):
    '''
    Generates rolling window statistical features (mean & standard deviation) for the target column.
    
    Rolling statistics compute aggregate values over a moving time window. These features help capture 
    long-term trends and short-term volatility.

    Parameters:
    -----------
    df: pandas DataFrame
        The input DataFrame containing the target column.
    target_col: str
        The name of the target column for which rolling statistics are computed.
    windows: list of int
        A list of window sizes for computing rolling statistics.

    Returns:
    --------
    df: pandas DataFrame
        The DataFrame with additional rolling mean and standard deviation columns.
    '''
    for window in windows:
        df[f"{target_col}_rolling_mean{window}"] = df[target_col].rolling(window=window).mean()
        df[f"{target_col}_rolling_std{window}"] = df[target_col].rolling(window=window).std()
    return df

def add_time_features(df):
    '''
    Extracts and adds time-based features from the 'observation_date' column.

    Time-based features help the model capture seasonal trends, economic cycles, 
    and other time-dependent variations in the data.

    Parameters:
    -----------
    df: pandas DataFrame
        The input DataFrame containing a datetime column named 'observation_date'.

    Returns:
    --------
    df: pandas DataFrame
        The DataFrame with additional time-based features:
        - 'year': Extracts the year from the date.
        - 'month': Extracts the month (1-12) from the date.
        - 'quarter': Extracts the quarter (1-4) from the date.
    '''
    df["year"] = df["observation_date"].dt.year
    df["month"] = df["observation_date"].dt.month
    df["quarter"] = df["observation_date"].dt.quarter
    return df

def create_sequences(data, exog, fft_real, fft_imag, target, seq_len, config):
    """
    Creates sequences dynamically based on the provided configuration.

    Parameters:
    -----------
    data: numpy array, time-series target data.
    exog: numpy array, exogenous variables (if any).
    fft_real: numpy array, real part of FFT transformation.
    fft_imag: numpy array, imaginary part of FFT transformation.
    target: numpy array, target variable.
    seq_len: int, sequence length.
    config: dict, configuration specifying which features to include.

    Returns:
    --------
    Processed sequences for model input.
    """
    X, X_exog, y = [], [], []
    for i in range(len(data) - seq_len):
        seq_data = data[i : i + seq_len].flatten()
        
        # Apply Fourier Transform if enabled in config
        if config.get("use_fft", False):
            seq_data = np.hstack([seq_data, fft_real[i], fft_imag[i]])  

        # Include exogenous variables if enabled
        if config.get("use_exog", False) and exog is not None:
            X_exog.append(exog[i])

        X.append(seq_data)
        y.append(target[i + seq_len])  

    if config.get("use_exog", False):
        return np.array(X), np.array(X_exog), np.array(y)  # Return exog separately
    else:
        return np.array(X), np.array(y)
        

def train_val_test_split(X : np.array, y : np.array, train_size : float, val_size : float, test_size=None):
    """
    Splits dataset into training, validation, and test sets.

    Parameters:
    -----------
    X: numpy array, time-series input data.
    y: numpy array, time-series target data.
    train_size: float giving the proportion of data represented by the training set.
    val_size: float giving the proportion of data represented by the validation set.
    train_size (optional): float giving the proportion of data represented by the test set. (inferred if not included)

    Returns:
    --------
    Training, validation, and test input and target datasets.
    """
    if not test_size:
        test_size = 1 - (train_size + val_size)
    if train_size + val_size + test_size != 1:
        raise ValueError('Train, validation, and test sizes must sum to 1.')

    train_idx = int(len(X) * train_size)
    valid_idx = int(len(X) * (train_size + val_size))

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_valid, y_valid = X[train_idx:valid_idx], y[train_idx:valid_idx]
    X_test, y_test = X[valid_idx:], y[valid_idx:]

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def load_data(train_file : str, sequence_length=48, config={}):
    '''
    Loads and preprocesses time-series data for training a machine learning model.

    This function:
    - Reads a CSV file containing time-series data.
    - Converts the date column into a datetime format.
    - Sorts data by date.
    - Creates lagged features, rolling statistics, and time-based features.
    - Splits data into training, validation, and test sets.
    - Applies feature scaling.
    - Optionally computes Fourier Transform features.
    - Formats data into sequences for use in time-series models.

    Parameters:
    -----------
    train_file: str
        The file path to the CSV containing time-series data.
    sequence_length: int, default=48
        The number of past time steps to use for each sequence in the model.
    config: dict, optional
        A configuration dictionary that specifies whether to include FFT features 
        and exogenous variables. Of the form:
        {"use_fft": bool, "use_exog": bool}

    Returns:
    --------
    X_train_seq, y_train_seq: numpy arrays
        Training sequences for the model.
    X_valid_seq, y_valid_seq: numpy arrays
        Validation sequences.
    X_test_seq, y_test_seq: numpy arrays
        Test sequences.
    observation_dates: pandas Series
        The dates corresponding to the test set predictions.
    scaler: MinMaxScaler or StandardScaler
        The scaler used to normalize the input data.
    y_scaler: MinMaxScaler or StandardScaler
        The scaler used to normalize the target variable.
    exog_scaler: StandardScaler (if applicable)
        The scaler for exogenous features, if they are included.
    '''

    df = pd.read_csv(train_file)
    df["observation_date"] = pd.to_datetime(df["observation_date"], format="%m/%Y")
    df = df.sort_values(by="observation_date").reset_index(drop=True)

    target_col = "fred_PCEPI"
    df = add_lagged_features(df, target_col)
    df = add_rolling_features(df, target_col)
    df = add_time_features(df)

    df.dropna(inplace=True)

    exog_cols = [col for col in df.columns if col not in ["observation_date", target_col]]
    data = df[[target_col]].values.astype(np.float32)
    exog_data = df[exog_cols].values.astype(np.float32) if config.get("use_exog", False) else None
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_val_test_split(data, data, 0.7, 0.15, 0.15)
    
    X_scaler = MinMaxScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_valid = X_scaler.transform(X_valid)
    X_test = X_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_valid = y_scaler.transform(y_valid.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))

    if config.get("use_fft", False):
        train_fft_real, train_fft_imag = apply_fft_per_sequence(X_train, sequence_length)
        valid_fft_real, valid_fft_imag = apply_fft_per_sequence(X_valid, sequence_length)
        test_fft_real, test_fft_imag = apply_fft_per_sequence(X_test, sequence_length)
    else:
        train_fft_real, train_fft_imag, valid_fft_real, valid_fft_imag, test_fft_real, test_fft_imag = [None for _ in range(6)]
    
    if config.get("use_exog", False):
        exog_scaler = StandardScaler()
        X_exog_train = exog_scaler.fit_transform(exog_data[:len(X_train)])
        X_exog_valid = exog_scaler.transform(exog_data[len(X_train):len(X_train) + len(X_valid)])
        X_exog_test = exog_scaler.transform(exog_data[len(X_train) + len(X_valid):])
    
        X_train_seq, X_exog_train_seq, y_train_seq = create_sequences(X_train, X_exog_train, train_fft_real, train_fft_imag, y_train, sequence_length, config)
        X_valid_seq, X_exog_valid_seq, y_valid_seq = create_sequences(X_valid, X_exog_valid, valid_fft_real, valid_fft_imag, y_valid, sequence_length, config)
        X_test_seq, X_exog_test_seq, y_test_seq = create_sequences(X_test, X_exog_test, test_fft_real, test_fft_imag, y_test, sequence_length, config)
    
        return X_train_seq, X_exog_train_seq, y_train_seq, X_valid_seq, X_exog_valid_seq, y_valid_seq, X_test_seq, X_exog_test_seq, y_test_seq, df["observation_date"].iloc[sequence_length:], exog_scaler, y_scaler
    
    X_train_seq, y_train_seq = create_sequences(X_train, None, train_fft_real, train_fft_imag, y_train, sequence_length, config)
    X_valid_seq, y_valid_seq = create_sequences(X_valid, None, valid_fft_real, valid_fft_imag, y_valid, sequence_length, config)
    X_test_seq, y_test_seq = create_sequences(X_test, None, test_fft_real, test_fft_imag, y_test, sequence_length, config)
    
    return X_train_seq, y_train_seq, X_valid_seq, y_valid_seq, X_test_seq, y_test_seq, df["observation_date"].iloc[sequence_length:], y_scaler

def prepare_dataloader(X : np.array, y : np.array, batch_size=32) -> DataLoader:
    '''
    Converts dataset into PyTorch DataLoader.

    Parameters:
    -----------
    X: numpy array, time-series input data.
    y: numpy array, time-series target data.
    batch_size: integer, size of the batches returned by the DataLoader.

    Returns:
    --------
    The DataLoader as requested
    '''
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)