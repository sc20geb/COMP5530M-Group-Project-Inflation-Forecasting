from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def makeModelComparison(modelNames, figsize=(12, 8), barWidth=0.2) -> plt:
    fig, ax = plt.subplots(figsize=figsize)

    #concatenate DataFrames from each model
    fullDf = pd.DataFrame()
    for name in modelNames:
        df = pd.read_csv(f'.\\Data\\{name}_PCEPI_eval.csv')
        fullDf = pd.concat([fullDf, df])
    fullDf = fullDf.drop('Unnamed: 0', axis=1)

    #create bar locations
    brs = [np.arange(len(modelNames))]
    for i in range(1, len(fullDf.columns)-1):
        brs.append([x + barWidth for x in brs[i-1]])

    #plot bars
    for i, col in enumerate(fullDf.columns[1:]):
        p = ax.bar(brs[i], fullDf[col], width=barWidth, label=col)
        ax.bar_label(p, label_type='center')
    #formatting
    plt.xticks([r + barWidth for r in range(len(modelNames))], fullDf['Model'])
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.legend()
    plt.title(f'Score Comparisons for {modelNames}')
    return plt