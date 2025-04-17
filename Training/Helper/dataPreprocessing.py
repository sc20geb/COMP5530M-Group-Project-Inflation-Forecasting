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
import re
from statsmodels.tsa.stattools import adfuller, ccf, grangercausalitytests, coint
from collections.abc import Callable
from math import prod

# Get absolute path to the project root (2 levels up from this file)
MODULE_PATH = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))


# Define paths to datasets
TRAIN_DATA_PATH_1990S = os.path.join(MODULE_PATH, 'Data', 'Train', 'train1990s.csv')
TRAIN_DATA_PATH_2000S = os.path.join(MODULE_PATH, 'Data', 'Train', 'train2000s.csv')
TEST_DATA_PATH_1990S  = os.path.join(MODULE_PATH, 'Data', 'Test', 'test1990s.csv')
TEST_DATA_PATH_2000S  = os.path.join(MODULE_PATH, 'Data', 'Test', 'test2000s.csv')

# Define standard training and validation split proportions
TRAIN_DATA_SPLIT = 0.8
VAL_DATA_SPLIT = 0.2


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def apply_fft_per_sequence(data : np.array, seq_len : int, num_components : int = 10):
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

def add_lagged_features(df : pd.DataFrame, target_cols : list[str] , lags : list[int] = [1, 3, 6]):
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

def add_rolling_features(df : pd.DataFrame, target_col : str, windows : list[int] = [3, 6, 12]):
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

def add_time_features(df : pd.DataFrame, date_col : str = 'observation_date'):
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

def add_modified_feature(df : pd.DataFrame , target_col : str, func : Callable[[pd.Series], pd.Series]):
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

def create_sequences(data : np.array, target : np.array, seq_len : int, exog : np.array = None, 
                     fft_real : np.array = None, fft_imag : np.array = None, config : dict = {'use_fft': False, 'use_exog': False}):
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


def prepare_dataloader(X : np.array, y : np.array, X_exog : np.array = None, shuffle : bool = True, batch_size : int = 32):
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

def drop_near_constant_cols(df : pd.DataFrame, threshold : float = 1e-6):
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


def sklearn_fit_transform(*args : list[pd.DataFrame, ], column_names : list[str] = None):
    """
    Fits and transforms the provided datasets using the provided sklearn function.
    Transform is fitted on the training (first) set, then applied to the training set followed by all other sets.

    Parameters:
    -----------
    *args: n arguments, consisting of the following items at the following indices:
        [0:n-1]: DataFrame objects, consisting of the following DataFrames at the following indices:
            0: DataFrame containing the training data.
            Other: DataFrames containing validation, test, or other data.
        n-1: sklearn transform object
            The transform to be applied to the data. Must implement the fit-transform sklearn API.
    column_names: str (optional)
        The names of the columns of the output df; must have len(column_names) == len(args[i].columns) for all i < n-1

    Returns:
    --------
    List of DataFrames containing their respective transformed datasets (in the same order as passed) and fitted scaler
    """
    sklearn_func = args[-1]
    sklearn_func.fit(args[0])  # fit on train only

    transformed_dfs = [sklearn_func.transform(df) for df in args[:-1]]

    if column_names: new_cols = column_names
    else: new_cols = [f"{type(sklearn_func).__name__}_{i+1}" for i in range(transformed_dfs[0].shape[1])]
    return [pd.DataFrame(transformed_dfs[i], index=df.index, columns=new_cols) for i, df in enumerate(args[:-1])], sklearn_func

def integer_index(dfs : list[pd.DataFrame] , start : int = 0):
    """
    Changes the index of the provided DataFrame(s) to be integers in the range [0, len(dataframe)].
    If a list of DataFrame objects is provided, returns a list of integer-index-converted dataframes.

    Parameters:
    -----------
    dfs: list[DataFrame] or DataFrame
        (potentially several) DataFrame(s) to be converted to an integer index.
    start: int
        Value from which to start the index.

    Returns:
    --------
    list[DataFrame] or DataFrame - the DataFrame(s) input with their index replaced with integers.
    """
    if type(dfs) != list: dfs = [dfs]
    returns = []
    for df in dfs:
        int_index = np.arange(start, len(df)+start)
        df_copy = df.copy()
        df_copy.index = int_index
        returns.append(df_copy)
    if len(returns) == 1: return returns[0]
    return returns

def difference2Cols(df:pd.DataFrame,col1:str,col2:str, n:int):
    '''
    This function differences 2 variables, to achieve stationarity

    Parameters:
    -----------
    col1: Name of the 1st column to be differenced.
    col2: Name of the 2nd column to be differenced.
    df: Dataframe which has the data for col1 and col2.
    n: The Order of differencing.
    Returns:
    ---------
    A pandas dataframe with 2 columns with the differenced data (removing the nulls).
    '''
    return df[[col1,col2]].diff(n).iloc[n:,:]

def make_stationary(df:pd.DataFrame,col1:str,col2:str ):

    '''
    This function makes 2 columns of a dataframe stationary, by taking the minimal nth difference which achieves
    stationarity (using the dickey-fuller stationarity test with a significance level of 5%).

    Parameters:
    -----------
    col1: Name of the 1st column, which needs to be transformed into a stationary time series.
    col2: Name of the 2nd column, which needs to be transformed into a stationary time series.
    df: Dataframe which has the data for col1 and col2.

    Returns:
    --------
    Returns a pandas dataframe with 2 columns of the stationary data. NOTE: if no stationarity is achieved, NaN is returned.
    '''
     # Check both variables are stationary:
    if adfuller(df[col1])[1]<0.05 and adfuller(df[col2])[1]<0.05:
        return df[[col1,col2]]

    for i in range(1,13):
        
        diffDf= difference2Cols(df,col1,col2,i)# Difference data to try achive stationarity
        # Check for stationarity:
        if adfuller(diffDf[col1])[1]<0.05 and adfuller(diffDf[col2])[1]<0.05:
            return diffDf
        
    # Return NaN if no differencing achieved stationarity within 12 iterations      
    return np.nan


def find_best_corr(df:pd.DataFrame,col2:str, target:str='fred_PCEPI', maxlag=12):

    '''
    This function finds the maximal cross correlation between target and col2, using 12 lags (year).
    NOTE: for reliable cross correlation results, the time series needs to be stationary, hence make_stationary is used.
    NOTE: if stationarity is NOT achieved, then NaN is returned.

    Parameters:
    -----------
    target: The name of the target column.
    col2: The name of the other column which is used to calculate the cross correlation between the target.
    df: Dataframe which has the data for target and col2.

    Returns:
    --------
    returns maximal cross correlation out of the 12 time lagged cross correlation values.
    '''
    # Make time-series stationary:
    stationary_df=make_stationary(df,target,col2)

    # Return NaN if stationarity is NOT achieved:
    if stationary_df is np.nan:
        return np.nan
    
    # Return maximal ccf value:
    return np.max(np.abs(ccf(stationary_df[col2],stationary_df[target],nlags=maxlag)))


def granger_causes(df:pd.DataFrame,col:str,target:str='fred_PCEPI'):
    '''
    Perfom granger causlaity test, where the following is tested: col granger-causes target, with a
    5% significance level.

    Parameters:
    -----------
    target: The name of the target column.
    col: The name of the other column which granger-causes target.
    df: Dataframe which has the data for target and col2.

    Returns:
    --------
    A boolean value, True if col granger cause target, else False.

    '''
    # calculate the p-values
    granger=grangercausalitytests(df[[target,col]],maxlag=12,verbose=False)

    # loop over all the lags:
    for key in granger.keys():
        # Check if it is significant using 5% significance level
        if granger[key][0]['ssr_chi2test'][1]<0.05:
            return True
    return False

def rank_features_ccf(df:pd.DataFrame, targetCol:str='fred_PCEPI',maxlag=12):

    '''
    This function ranks the exogenous variables by ccf value.

    Parameters:
    -----------
    df: pandas dataframe containing all the exogenous data and target data.


    Returns:
    --------
    Returns pandas dataframe where the columns are ordered by ccf value in descending order.
    '''


    #Calculate cross correlation values:
    ccf_cols= np.array((df.columns.copy().drop(targetCol)))

    corrs=[]
    for exog in ccf_cols:
        x= find_best_corr(df,exog,targetCol, maxlag=maxlag)
        if x is np.nan:
            x=0.
        corrs.append(x)
    
    best_corrs=np.argsort(corrs)[::-1] 
    ccf_cols=ccf_cols[best_corrs]
    
    return ccf_cols

def get_untransformed_exog(df:pd.DataFrame):
    '''
    This function returns a dataframe without the transformed exogenous variables from the fred dataset (if there are too many features).

    Parameters:
    -----------
    df: pandas dataframe containing all exogenous variables and target variable

    Returns:
    --------
    Returns a pandas dataframe with the transformed fred variables removed.
    '''
    transformedCols=[]
    for i in df.columns:
        match=re.findall(r'fred_.*_.*',i)
        if match!=[]:
            transformedCols.append(match[0])

    return df.drop(transformedCols,axis=1)

def add_dimension(arr : np.array):
    '''
    Adds a single dimension to the end of a numpy array.

    Parameters:
    -----------
    arr: numpy array
        Array to which the new dimension should be added.

    Returns:
    --------
    Numpy array with added dimension
    '''
    return arr.reshape(*arr.shape, 1)

# Random Forest builder func
def build_feature_matrix(
    df: pd.DataFrame,
    *,
    target_col: str,
    n_lags: int,
    exog_cols: list[str],
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create lag + exogenous feature matrix.

    Parameters
    ----------
    df : DataFrame with at least `target_col` and `exog_cols`
    target_col : str  – column to forecast
    n_lags : int      – number of past lags to add
    exog_cols : list[str] – extra regressors
    drop_na : bool    – if True, drop rows that still have NaN after lagging

    Returns
    -------
    X, y : DataFrame, Series
    """
    df = df.copy()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    feature_cols = [f"lag_{lag}" for lag in range(1, n_lags + 1)] + exog_cols

    if drop_na:
        df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    return df[feature_cols], df[target_col]
