import numpy as np

from python_helper import log


def getBestFitList(targetData, trainningData, dataSet, neighbors):
    log.debug(getBestFitList, f'Querying {neighbors} samples ...')
    targetArray = np.asarray(targetData)
    dataSetMatrix = np.asarray([*trainningData])
    euclidianDistanceList = np.asarray(np.sum((targetArray - dataSetMatrix)**2, axis=1))

    log.debug(getBestFitList, f'euclidianDistanceList = {euclidianDistanceList}')
    bestFitIndexList = np.argsort(euclidianDistanceList)
    log.debug(getBestFitList, f'bestFitIndexList = {bestFitIndexList}')
    log.debug(getBestFitList, f'dataSetMatrix = {dataSetMatrix}')

    bestFitList = [dataSet[bestFitIndex] for bestFitIndex in bestFitIndexList[:neighbors]]
    log.debug(getBestFitList,f'Optimum match: {bestFitList}')
    return bestFitList, bestFitIndexList[:neighbors]


def getPredictions(Y, X, trainningData, neighbors):
    assert len(X) == len(trainningData.values), f'{X} ---- {trainningData.values} ---- {len(X)} == {len(trainningData.values)}'
    predictionList = []
    for targetIndex, target in enumerate(Y):
        if targetIndex % 100 == 0:
            log.status(getPredictions, f'Currently predicting the {targetIndex}° test case')
        total = 0
        bestFitList, bestFitIndexList = getBestFitList(target, X, [*trainningData.values], neighbors)
        for hit in bestFitList:
            total += hit[-1]
        predictionList.append(total / len(bestFitList))
    return predictionList
