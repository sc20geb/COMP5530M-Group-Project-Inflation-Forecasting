from datetime import datetime, timedelta

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

def str_to_float(x):
    '''
    This function converts a string into a float, where the original data has K,M,B as shorthand instead of writing the zeros (investing.com often has data of this form).
    
    Parameters:
    -----------
    x: string of format xY, where Y is a character and x is some float

    Returns:
    --------
    float

    '''
    #If input is already a float then return the input:
    if type(x)== float:
        return x
    # convert K to 1000's
    if x[-1]=='K':
        return float(x[:-1])*1e3
    # convert M to millions's
    elif x[-1]=='M':
        return float(x[:-1])*1e6
    #Convert B to billions
    elif x[-1]=='B':
        return float(x[:-1])*1e9
    #Return converted number
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