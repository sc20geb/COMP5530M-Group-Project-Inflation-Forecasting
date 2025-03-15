#TODO: Discuss where this should be in the directory system: 
# Either here, where it is most likely to be used in notebooks since these notebooks produce these weights, or in Models/Weights/, or in Models/?
import os

def cleanWeightFiles(modelName : str, earlyStopped:bool = True, filePath: str = os.path.join('..', 'Models', 'Weights')):
    """
    Removes files containing a specified model name from a directory. (USE CAREFULLY)
    By default, removes early-stopping weight files for a specified model from the Models/Weights/ directory.

    Parameters:
    -----------
    modelName: string, name of the model used in the file names.
    earlyStopped (optional): boolean, whether to delete only early-stopping files.
    filePath (optional): string or bytes (interpretable as file path), explicit path to directory containing files to be deleted.

    Returns:
    --------
    void
    """
    for fileName in os.listdir(filePath):
        if modelName not in fileName: continue
        if earlyStopped and 'BEST_STOPPED_AT' in fileName:
            os.remove(os.path.join(filePath, fileName))
        else:
            os.remove(os.path.join(filePath, fileName))