from datetime import datetime, timedelta

#rounds datetime up or down to the nearest year
def round_to_year(dt):
    try:
        dt.replace(month=2, day=29)
        leap = True
    except:
        leap = False
    return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0) + timedelta(days=365+leap if dt.month >= 7 else 0)

#gives test date matching the requested test set ratio of the range given (to the nearest year)
def getTestDate(testRatio : float, startYear : int, endYear : int, verbose=False):
    if testRatio < 0 or testRatio > 1: raise ValueError('Test ratio must be between 0 and 1.')
    if endYear <= startYear: raise ValueError('End year must be later than start year.')
    
    startYearDatetime = datetime.strptime(str(startYear), '%Y')
    endYearDatetime = datetime.strptime(str(endYear), '%Y')
    nTDDT = endYearDatetime - (testRatio * (endYearDatetime - startYearDatetime))
    rNTDDT = round_to_year(nTDDT)
    actualRatio = (endYearDatetime - rNTDDT) / (endYearDatetime - startYearDatetime)
    if actualRatio != testRatio and verbose: print(f'Rounding to years changed ratios. New ratios are:\nTest ratio: {actualRatio}, Train ratio: {1-actualRatio}')
    return rNTDDT.strftime('%m/%Y')