#TODO: Discuss where this should be in the directory system: 
# Either here, where it is most likely to be used in notebooks since these notebooks produce these weights, or in Models/Weights/, or in Models/?
import os
from glob import glob
from pathlib import Path

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
    dirPath = Path(dirPath)
    for filePath in dirPath.glob('*.pth'):
        fileName = filePath.name
        if modelName not in fileName: continue
        if earlyStopped:
            if 'BEST_STOPPED_AT' in fileName:
                os.remove(os.path.join(dirPath, fileName))
                if verbose: print(f'Removed file at: {os.path.join(dirPath, fileName)}')
            else: continue
        else:
            os.remove(os.path.join(dirPath, fileName))
            if verbose: print(f'Removed file at: {os.path.join(dirPath, fileName)}')