import torch
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.stattools import pacf
import statsmodels.api as sm

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

def add_lagged_features(df, target_cols, lags=[1, 3, 6]):
    '''
    Generates lagged features for the given target column(s).
    
    Lagging means shifting the values of the target column by a certain number of time steps 
    to create past observations as features. This helps the model learn from previous values.

    Parameters:
    -----------
    df: pandas DataFrame
        The input DataFrame containing the target column.
    target_col: list[str]
        The name(s) of the target column(s) for which lag features are created.
    lags: list of int
        A list of lag values indicating how many time steps back to shift the data.

    Returns:
    --------
    df: pandas DataFrame
        The DataFrame with additional lagged columns.
    '''
    for t_col in target_cols:
        for lag in lags:
            df[f"{t_col}_lag{lag}"] = df[t_col].shift(lag)
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

def add_modified_feature(df, target_col, func):
    '''
    Adds a new column to the DataFrame provided representing the target column with some function applied.

    Parameters:
    -----------
    df: pandas DataFrame
        The input DataFrame containing the target column
    target_col: str
        The name of the target column used to create the new modified feature.
    func: Callable[pd.Series]
        Function taking a pandas series as input, which returns a modified version of that series.

    Returns:
    --------
    df: pandas DataFrame
        The DataFrame with the additional feature specified, named according to the __name__ attribute of the function provided and the name of the target column.
    '''
    if target_col not in df.columns: raise ValueError(f'{target_col} not a column in the input DataFrame.')
    df[f"{func.__name__}_{target_col}"] = func(df[target_col])
    return df

def create_sequences(data, target, seq_len, exog=None, fft_real=None, fft_imag=None, config={'use_fft': False, 'use_exog': False}):
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
        if config.get("use_fft", False) and fft_real is not None and fft_imag is not None:
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

def load_data(train_file : str, sequence_length=48, train_size : float = 0.7, val_size : float = 0.15, test_size : float = 0.15, config={}):
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
    train_size: float,
        The proportion of the dataset assigned to training.
    val_size: float,
        The proportion of the dataset assigned to validation.
    test_size: float,
        The proportion of the dataset assigned to testing.
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
    df = add_lagged_features(df, [target_col])
    df = add_rolling_features(df, target_col)
    df = add_time_features(df)

    df.dropna(inplace=True)

    exog_cols = [col for col in df.columns if col not in ["observation_date", target_col]]
    data = df[[target_col]].values.astype(np.float32)
    exog_data = df[exog_cols].values.astype(np.float32) if config.get("use_exog", False) else None
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_val_test_split(data, data, train_size, val_size, test_size)
    
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
    
        X_train_seq, X_exog_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length, exog=X_exog_train, fft_real=train_fft_real, fft_imag=train_fft_imag, config=config)
        X_valid_seq, X_exog_valid_seq, y_valid_seq = create_sequences(X_valid, y_valid, sequence_length, exog=X_exog_valid, fft_real=valid_fft_real, fft_imag=valid_fft_imag, config=config)
        X_test_seq, X_exog_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length, exog=X_exog_test, fft_real=test_fft_real, fft_imag=test_fft_imag, config=config)
    
        return X_train_seq, X_exog_train_seq, y_train_seq, \
               X_valid_seq, X_exog_valid_seq, y_valid_seq, \
               X_test_seq, X_exog_test_seq, y_test_seq, \
               df["observation_date"].iloc[sequence_length:], X_scaler, exog_scaler, y_scaler

    # If no exogenous variables exist, return only X and y
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length, fft_real=train_fft_real, fft_imag=train_fft_imag, config=config)
    X_valid_seq, y_valid_seq = create_sequences(X_valid, y_valid, sequence_length, fft_real=valid_fft_real, fft_imag=valid_fft_imag, config=config)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length, fft_real=test_fft_real, fft_imag=test_fft_imag, config=config)
        
    return X_train_seq, y_train_seq, \
        X_valid_seq, y_valid_seq, \
        X_test_seq, y_test_seq, \
        df["observation_date"].iloc[sequence_length:], y_scaler


def prepare_dataloader(X, y, X_exog=None, shuffle=True, batch_size=32):
    """
    Converts dataset, optionally with exogenous variables, into PyTorch DataLoader.

    Parameters:
    -----------
    X: numpy array, time-series input data.
    y: numpy array, time-series target data.
    X_exog (optional): numpy array, exogenous variable input.
    shuffle (optional): boolean, shuffle argument for DataLoader.
    batch_size (optional): int, size of batches.

    Returns:
    --------
    PyTorch DataLoader.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    if X_exog is not None:
        X_exog_tensor = torch.tensor(X_exog, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, X_exog_tensor, y_tensor)
    else:
        dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def best_lag_selection(train_series : sm.tools.typing.ArrayLike1D, max_lags : int = 12, verbose: bool = False) -> int:
    """
    Picks a 'best_lag' for ARDL in levels by:
      1) Checking partial autocorrelation (PACF)
      2) Checking a simple AR(lag) model's AIC

    Parameters:
    -----------
    train_series: 1D array-like
        Series containing the training target data.
    max_lags: integer
        The maximum number of lags to be checked with the AR model.
    verbose: boolean
        Whether or not the best lags found by partial autocorrelation and the AR model respectively are printed.

    Returns:
    --------
    Integer best lag.
    """
    pacf_vals = pacf(train_series, nlags=max_lags)
    best_pacf_lag = np.argmax(np.abs(pacf_vals[1:])) + 1

    best_aic_lag, best_aic = 1, float("inf")
    for lag in range(1, max_lags + 1):
        y = train_series[lag:]
        X = train_series.shift(lag)[lag:]
        X = sm.add_constant(X, prepend=True)
        try:
            model = sm.OLS(y, X).fit()
            if model.aic < best_aic:
                best_aic_lag = lag
                best_aic = model.aic
        except:
            continue

    selected_lag = min(best_pacf_lag, best_aic_lag)
    if verbose: print(f" - best_lag by PACF: {best_pacf_lag}, best_lag by AIC: {best_aic_lag}")
    return selected_lag

def drop_near_constant_cols(df, threshold=1e-6):
    """
    Removes near-constant columns according to their standard deviation (for models that struggle with constant values).

    Parameters:
    -----------
    df: DataFrame
        DataFrame from which near-constant columns are removed
    threshold: float
        Minimum standard deviation for a column to be considered non-constant

    Returns:
    --------
    Modified DataFrame, list of names of dropped columns
    """
    orig_cols = df.columns
    newDf = df.loc[:, df.std() > threshold]
    return newDf, list(set(orig_cols)-set(newDf.columns))


def sklearn_fit_transform(train_df, test_df, sklearn_func, **func_kwargs):
    """
    Creates DataFrames containing sklearn-transformed train and test datasets.
    Transform is fitted on the training set, and applied to both the training and testing sets.

    Parameters:
    -----------
    train_df: DataFrame
        DataFrame containing the training data
    test_df: DataFrame
        DataFrame containing the test data
    sklearn_func: sklearn transform
        The transform to be applied to the data. Must implement the fit-transform sklearn API
    **func_kwargs: keyword arguments
        Keyword arguments for the sklearn function

    Returns:
    --------
    DataFrames containing the transformed training dataset and the transformed test dataset respectively.
    """
    obj = sklearn_func(**func_kwargs)
    obj.fit(train_df)  # fit on train only

    transformed_train = obj.transform(train_df)
    transformed_test = obj.transform(test_df)

    new_cols = [f"{sklearn_func.__name__}_{i+1}" for i in range(transformed_train.shape[1])]
    return pd.DataFrame(transformed_train, index=train_df.index, columns=new_cols), pd.DataFrame(transformed_test,  index=test_df.index,  columns=new_cols)

def integer_index(dfs):
    """
    Changes the index of the provided DataFrame(s) to be integers in the range [0, len(dataframe)].
    If a list of DataFrame objects is provided, returns a list of integer-index-converted dataframes.

    Parameters:
    -----------
    dfs: list[DataFrame] or DataFrame
        (potentially several) DataFrame(s) to be converted to an integer index.

    Returns:
    --------
    list[DataFrame] or DataFrame - the DataFrame(s) input with their index replaced with integers.
    """
    if type(dfs) != list: dfs = [dfs]
    returns = []
    for df in dfs:
        int_index = np.arange(len(df))
        df_copy = df.copy()
        df_copy.index = int_index
        returns.append(df_copy)
    if len(returns) == 1: return returns[0]
    return returns