import torch
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.stattools import pacf
import statsmodels.api as sm
import logging
import os

# Useful 'macros' (global variables) defining paths to train and test data 
MODULE_PATH = os.path.abspath(os.path.join('..'))
TRAIN_DATA_PATH_1990S = os.path.join(MODULE_PATH, 'Data', 'Train', 'train1990s.csv')
TRAIN_DATA_PATH_2000S = os.path.join(MODULE_PATH, 'Data', 'Train', 'train2000s.csv')
TEST_DATA_PATH_1990S = os.path.join(MODULE_PATH, 'Data', 'Test', 'test1990s.csv')
TEST_DATA_PATH_1990S = os.path.join(MODULE_PATH, 'Data', 'Test', 'test2000s.csv')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def minMaxScale(vals: np.array) -> np.array:
    """
    Applies Min-Max scaling to normalize the input array between 0 and 1.

    Parameters:
    -----------
    vals: numpy array
        The input array to be scaled.

    Returns:
    --------
    numpy array
        The min-max scaled version of the input array.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(vals.reshape(-1, 1)).flatten()

def apply_fft_per_sequence(data, seq_len, num_components=10):
    """
    Computes the Fast Fourier Transform (FFT) for each sequence in the input time-series data 
    and extracts the real and imaginary components.

    Parameters:
    -----------
    data: numpy array
        The input time-series data, where each row represents a timestamp.
    seq_len: int
        The length of each sequence for which FFT is applied, determining how many past 
        values are considered in the transformation.
    num_components: int, default=10
        The number of FFT components to retain from both the real and imaginary parts.

    Returns:
    --------
    fft_real: numpy array
        Log-transformed absolute values of the real components of the FFT.
    fft_imag: numpy array
        Log-transformed absolute values of the imaginary components of the FFT.
    """
    def compute_fft(seq):
        transformed = fft(seq.flatten())
        return np.hstack([
            np.log1p(np.abs(np.real(transformed[:num_components]))),
            np.log1p(np.abs(np.imag(transformed[:num_components])))
        ])
    
    fft_features = np.apply_along_axis(compute_fft, axis=1, arr=np.lib.stride_tricks.sliding_window_view(data, (seq_len, data.shape[1])))
    return fft_features[:, :num_components], fft_features[:, num_components:]

def add_lagged_features(df, target_cols, lags=[1, 3, 6]):
    """
    Creates lagged features for the specified target column(s) by shifting values 
    backward in time, allowing the model to incorporate past observations as input.

    Parameters:
    -----------
    df: pandas DataFrame
        The input DataFrame containing the target column(s).
    target_cols: list[str]
        A list of target column names for which lag features should be created.
    lags: list of int, default=[1, 3, 6]
        A list of lag values specifying how many time steps back to shift the data.

    Returns:
    --------
    pandas DataFrame
        The original DataFrame with additional lagged feature columns.
    """
    lagged_data = {f"{t_col}_lag{lag}": df[t_col].shift(lag) for t_col in target_cols for lag in lags}
    return df.assign(**lagged_data)

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

def add_time_features(df, date_col: str = 'observation_date'):
    """
    Extracts and adds time-based features from the 'observation_date' column to 
    enhance the model's ability to capture seasonal patterns, economic cycles, 
    and other time-dependent variations.

    Parameters:
    -----------
    df: pandas DataFrame
        The input DataFrame containing a datetime column named 'observation_date'.
    date_col: str
        The name of the column in df containing the datetime objects from which time features are to be extracted.

    Returns:
    --------
    pandas DataFrame
        The original DataFrame with additional time-based features:
        - 'year': The year extracted from the date.
        - 'month': The month (1-12) extracted from the date.
        - 'quarter': The quarter (1-4) extracted from the date.

    Raises:
    -------
    KeyError:
        If the 'observation_date' column is missing from the DataFrame.
    """
    if date_col not in df:
        logging.error(f"Missing '{date_col}' column in DataFrame.")
        raise KeyError(f"Missing '{date_col}' column in DataFrame.")
    
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    
    logging.info(f"Added time features: year, month, quarter. DataFrame shape: {df.shape}")
    
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
    Creates sequences of a specified length from the input and target data dynamically based on the provided configuration.

    Parameters:
    -----------
    data: numpy array
        Time-series target data.
    target: numpy array
        Target variable.
    seq_len: int
        Sequence length.
    exog: numpy array (optional)
        Exogenous variables (if any).
    fft_real: numpy array (optional)
        Real part of FFT transformation.
    fft_imag: numpy array (optional)
        Imaginary part of FFT transformation.
    config: dict (optional)
        Configuration specifying which features to include.

    Returns:
    --------
    Processed sequences for model input.
    """
    if data.ndim == 1:  
        data = data.reshape(-1, 1)  # Convert to 2D

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
        return np.array(X), np.array(X_exog), np.array(y)
    else:
        return np.array(X), np.array(y)

        
def train_val_test_split(X, y, train_size=0.7, val_size=0.15, test_size=None):
    """
    Splits a dataset into training, validation, and test sets while ensuring 
    the specified proportions sum to 1.

    Parameters:
    -----------
    X: numpy array
        The time-series input data.
    y: numpy array
        The corresponding target variable data.
    train_size: float, default=0.7
        The proportion of the dataset assigned to training.
    val_size: float, default=0.15
        The proportion of the dataset assigned to validation.
    test_size: float, optional
        The proportion of the dataset assigned to testing. If not provided, 
        it is inferred as `1 - (train_size + val_size)`.

    Returns:
    --------
    tuple:
        - X_train (numpy array): Training input data.
        - y_train (numpy array): Training target data.
        - X_valid (numpy array): Validation input data.
        - y_valid (numpy array): Validation target data.
        - X_test (numpy array): Test input data.
        - y_test (numpy array): Test target data.

    Raises:
    -------
    ValueError:
        If the sum of `train_size`, `val_size`, and `test_size` does not equal 1.
    """
    if test_size is None:
        test_size = 1 - train_size - val_size
    if not np.isclose(train_size + val_size + test_size, 1, atol=1e-6):
        raise ValueError(f"Train, validation, and test sizes must sum to 1, got {train_size + val_size + test_size}")
    
    train_idx = int(len(X) * train_size)
    val_idx = int(len(X) * (train_size + val_size))

    return X[:train_idx], y[:train_idx], X[train_idx:val_idx], y[train_idx:val_idx], X[val_idx:], y[val_idx:]

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


def sklearn_fit_transform(*args, **func_kwargs):
    """
    Fits and transforms the provided datasets using the provided sklearn function.
    Transform is fitted on the training (first) set, then applied to the training set followed by all other sets.

    Parameters:
    -----------
    *args: n arguments, consisting of the following items at the following indices:
        [0:n-1]: DataFrame objects, consisting of the following DataFrames at the following indices:
            0: DataFrame containing the training data.
            Other: DataFrames containing validation, test, or other data.
        n-1: sklearn transform
            The transform to be applied to the data. Must implement the fit-transform sklearn API.
    **func_kwargs: keyword arguments
        Keyword arguments for the sklearn function

    Returns:
    --------
    List of DataFrames containing their respective transformed datasets (in the same order as passed).
    """
    sklearn_func = args[-1]
    obj = sklearn_func(**func_kwargs)
    obj.fit(args[0])  # fit on train only

    transformed_dfs = [obj.transform(df) for df in args[:-1]]

    new_cols = [f"{sklearn_func.__name__}_{i+1}" for i in range(transformed_dfs[0].shape[1])]
    return [pd.DataFrame(transformed_dfs[i], index=df.index, columns=new_cols) for i, df in enumerate(args[:-1])]

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