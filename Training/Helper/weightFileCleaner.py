#TODO: Discuss where this should be in the directory system: 
# Either here, where it is most likely to be used in notebooks since these notebooks produce these weights, or in Models/Weights/, or in Models/?
import os
from glob import glob
from pathlib import Path
from re import fullmatch

def cleanWeightFiles(modelName : str, earlyStopped:bool = True, dirPath: str = os.path.join('.'), verbose=False):
    """
    Removes files containing a specified model name from a directory. (USE CAREFULLY)
    By default, removes early-stopping weight files for a specified model from the Models/Weights/ directory.

    Parameters:
    -----------
    modelName: string, name of the model used in the file names.
    earlyStopped (optional): boolean, whether to delete only early-stopping files.
    dirPath (optional): string or bytes (interpretable as file path), explicit path to directory containing files to be deleted.
    verbose (optional): boolean, whether or not to print which files are being deleted

    Returns:
    --------
    void
    """
    # Captures entire filename depending on whether should only delete early-stopped or not
    pattern = f'{modelName}_BEST_STOPPED_AT_\\d+\\.pth' if earlyStopped else f'{modelName}.*\\.pth'

    # If finds a match to the regex specified above, remove the file (and inform the user of which file if verbose)
    dirPath = Path(dirPath)
    for filePath in dirPath.glob('*.pth'):
        fileName = filePath.name
        match = fullmatch(pattern, fileName)
        if match: 
            os.remove(os.path.join(dirPath, fileName))
            if verbose: print(f'Removed file at: {os.path.join(dirPath, fileName)}')