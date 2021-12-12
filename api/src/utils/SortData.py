import pandas as pd
import statistics
from functools import reduce

from python_helper import log, ObjectHelper

import DataUtils, Constants


def getItSorted(dataFrame: pd.DataFrame, columnName: str, targetColumnName: str):
    categories = {}
    for entry in dataFrame[[columnName, targetColumnName]].values:
        if entry[0] not in categories:
            categories[entry[0]] = [entry[-1]]
        else:
            categories[entry[0]].append(entry[-1])
    # categories[Constants.MEAN] = statistics.mean([collection for collection in categories.values if ObjectHelper.isCollection(collection)])
    categories[Constants.MEAN] = statistics.mean(DataUtils.mergeMultipleLists([k for k in categories.values() if ObjectHelper.isCollection(k)]))
    for cn,vs in {**{k:v for k,v in categories.items() if ObjectHelper.isCollection(v)}}.items():
        categories[cn] = statistics.mean(vs)
    maxValue = max(categories.values())
    for k,mean in {**categories}.items():
        categories[k] = mean / maxValue
    return sortIt(dataFrame, [columnName], {columnName:categories}), categories

def sortIt(dataFrame: pd.DataFrame, columnNames: [str], categoriesMap: dict):
    # log.prettyPython(sortIt, f'categoriesMap', categoriesMap, logLevel=log.STATUS)
    for columnName in columnNames:
        categories = categoriesMap.get(columnName, {})
        # dataFrame = DataUtils.replaceData(dataFrame, columnName, [*categories.keys()], [(v if (v > 0 and v < 1) else (0.0 if v <= 0 else 1.0)) for v in categories.values()])
        dataFrame = DataUtils.replaceData(dataFrame, columnName, [*categories.keys()], [*categories.values()])
        for value in dataFrame.copy()[[columnName]].values:
            if value > 1:
                dataFrame = DataUtils.replaceData(dataFrame, columnName, value, DataUtils.sigmoid(categories.get(Constants.MEAN) - (value / categories.get(Constants.MAX, value))))
                # dataFrame = DataUtils.replaceData(dataFrame, columnName, value, value / categories.get(Constants.MAX, 1.5 * value))
        # print(dataFrame)
    return dataFrame
