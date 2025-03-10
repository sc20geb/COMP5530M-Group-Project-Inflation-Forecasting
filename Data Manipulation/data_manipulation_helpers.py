from datetime import datetime, timedelta
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import pandas as pd

#
def round_to_year(dt : datetime) -> datetime:
    '''
    Rounds datetime up or down to the nearest year.
    
    Parameters:
    -----------
    dt: datetime object to be rounded

    Returns:
    --------
    datetime
    '''
    try:
        dt.replace(month=2, day=29)
        leap = True
    except:
        leap = False
    return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0) + timedelta(days=365+leap if dt.month >= 7 else 0)

#
def getTestDate(testRatio : float, startYear : int, endYear : int, verbose=False) -> str:
    '''
    Gives test date matching the requested test set ratio of the range given (to the nearest year).
    
    Parameters:
    -----------
    testRatio: Ratio of the year range desired to contain the test set
    startYear: First year in the range
    endYear: Final year in the range
    verbose: Toggles whether an output stream warning is given when the desired ratio is not achieved exactly

    Returns:
    --------
    string
    '''

    if testRatio < 0 or testRatio > 1: raise ValueError('Test ratio must be between 0 and 1.')
    if endYear <= startYear: raise ValueError('End year must be later than start year.')
    
    startYearDatetime = datetime.strptime(str(startYear), '%Y')
    endYearDatetime = datetime.strptime(str(endYear), '%Y')
    nTDDT = endYearDatetime - (testRatio * (endYearDatetime - startYearDatetime))
    rNTDDT = round_to_year(nTDDT)
    actualRatio = (endYearDatetime - rNTDDT) / (endYearDatetime - startYearDatetime)
    if actualRatio != testRatio and verbose: print(f'Rounding to years changed ratios. New ratios are:\nTest ratio: {actualRatio}, Train ratio: {1-actualRatio}')
    return rNTDDT.strftime('%m/%Y')

def str_to_float(x : str) -> float:
    '''
    This function converts a string into a float, where the original data has either
    K,M,B as shorthand instead of writing the zeros (investing.com often has data of this form), or
    contains commas, which are removed.
    
    Parameters:
    -----------
    x: string of format xY, where Y is a character and x is some float

    Returns:
    --------
    float

    '''

    shorthandMap = {'K' : 1e3, 'M' : 1e6, 'B' : 1e9}
    #Don't need to do anything if already a float
    if type(x) == float: return x
    #Remove the commas if they exist
    if ',' in str(x):
        x = x.replace(',', '')
    #Multiply by respective order of magnitude if needed
    if str(x)[-1] in list(shorthandMap.keys()):
        x = float(x[:-1])*shorthandMap[x[-1]]
    return float(x)

def rename_cols(df, name):
    '''
    This function renames the columns of a dataframe of the format "name"_"columnName"

    Parameters:
    -----------
    df: Pandas dataFrame which needs renaming of its columnns
    name: name to use for renaming

    Returns:
    --------
    void

    '''

    df.columns= list(map(lambda x: f'{name}_'+x,df))

def remove_percent(x):
    '''
    This function removes the percent sign from a string (representing a percentage) and converts it into a float.

    Parameters:
    -----------
    x: string representatation of a percentage to be converted to a float

    Returns:
    --------
    float without percentage sign

    '''
    return float(x[:-1])

#gets the number of decimal places in each column in the dataframe (assuming there are no null-columns)
def getColsDecimalPlaces(df : pd.DataFrame) -> dict:
    return {col : getColDecimalPlaces(df, col) for col in df.columns}

#gets the number of decimal places used in the first element of a column in the data frame provided
def getColDecimalPlaces(df : pd.DataFrame, colName : str) -> int:
    numString = str(df[colName][df[colName].isna() == False].iloc[0])
    splitNum = numString.split('.')
    if len(splitNum) != 2: raise ValueError(f'First extant element of column {colName} contains too many/ few decimal-point-separated sections.')
    return len(splitNum[1])

#interpolates the specified column and rounds to its original no. of decimal places
def interpolateAndRoundColumn(df : pd.DataFrame, colName : str, method='linear') -> pd.DataFrame:
    df[colName] = df[colName].interpolate(method=method)
    numDps = getColDecimalPlaces(df, colName)
    df[colName] = df[colName].round(decimals=numDps)
    return df

def getMissingCols(df, prin=False):
    missingCols = {}
    for i, value in enumerate(df.isna().sum()):
        if value != 0:
            missingCols[df.columns[i]] = value
            if prin: print(df.columns[i], value)
    return missingCols

def printMissingCols(df):
    if not getMissingCols(df, prin=True): print('There are no columns with missing values in the dataframe provided.')

def is_granger_caused(feature, y_trainDf, X_trainDf):

    '''
    This function returns True if feature granger causes target.

    Parameters:
    -----------
    feature: the name of the column within X_trainDf to test if it causes y_trainDf.

    Returns:
    --------

    True: if feature granger causes target
    False: if feature does NOT granger cause target.
    '''

    df_cpy= pd.concat((y_trainDf,X_trainDf[feature]),axis=1)# creates a deepcopy


    i=1 # initialized to 1 as PCEPI is not sationary with no preprocessing

    # peform adfuller test of xth differences untill both time series are stationary
    while adfuller(df_cpy.diff(i).dropna().iloc[:,0])[1]>0.05 or adfuller(df_cpy.diff(i).dropna().iloc[:,1])[1]>0.05:
        
        i+=1

        # Fail safe by keeping exogenous variable incase it by cause target, but may have a complex relationship
        if i>4:
            return True
        
    #Perform granger test:
    test=grangercausalitytests(df_cpy.diff(i).dropna(),maxlag=6,verbose=False)

    # see if there is  a time-lag which is granger caused:
    for key in test:
        
        #Return true if granger caused
        if test[key][0]['ssr_chi2test'][1]<0.05:
            return True
    
    # Else return false NOT granger caused)
    return False