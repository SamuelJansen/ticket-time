import datetime
import math
from functools import reduce
import statistics

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt

from python_helper import Constant as c
from python_helper import EnvironmentHelper, StringHelper, log, ObjectHelper
import globals

import Constants


PLOT_KINDS = ['scatter', 'line', 'bar']


def getFileNameWithPath(fileName):
    return StringHelper.join(['api', 'resource', 'data', fileName], EnvironmentHelper.OS_SEPARATOR)


def readCsv(csvFileName):
    return pd.read_csv(getFileNameWithPath(csvFileName))


def toCsv(content, csvFileName):
    # compression_opts = dict(method='zip', archive_name=getFileNameWithPath(csvFileName))
    # content.to_csv(csvFileName, index=False, compression=compression_opts)
    content.to_csv(getFileNameWithPath(csvFileName), index=False)


def plot(dataFrame, columns, title, kind):
    log.debug(plot, f'dataFrame: {dataFrame[[*columns.values()]]}')
    dataFrame.plot(x=columns.get('x'), y=columns.get('y'), kind=kind)
    plt.title(title)
    plt.show(block=False)


def showData(dataFrame, title):
    # from pandas.plotting import radviz
    # plt.figure()
    # radviz(dataFrame[['ID_GRUPO_SERVICO', 'TEMPOMINUTO', 'ID_AREA']], 'ID_AREA')
    ###- https://pandas.pydata.org/pandas-docs/version/0.9.1/visualization.html
    try:
        scatter_matrix(dataFrame, alpha=0.2, figsize=(8, 8), diagonal='kde')
        plt.title(title)
        plt.show(block=False)
    except Exception as e:
        log.failure(dataAnalysis, f'Not possible to show data of:{c.NEW_LINE}{dataFrame}', exception=e)
        plot(dataFrame, {'x': dataFrame.keys()[0], 'y': dataFrame.keys()[1]}, title, 'scatter')


def dataAnalysis(csvFileName, columnsToAnalyse, analysisBasedOnColumn=None, skipPlots=False):
    if analysisBasedOnColumn:
        dataFrameColumns = {**{k:None for k in columnsToAnalyse.keys()}, **{analysisBasedOnColumn:None}}
    else:
        dataFrameColumns = {k:None for k in columnsToAnalyse.keys()}
    log.prettyPython(dataAnalysis, 'dataFrameColumns', dataFrameColumns, logLevel=log.DEBUG)
    dataFrame, maxMap = getRawDataHandled(
        readCsv(csvFileName),
        dataFrameColumns,
        normalizeIt=False
    )
    showData(dataFrame, csvFileName)
    if not skipPlots:
        for columnToAnalyse in columnsToAnalyse:
            try:
                # plot(dataFrame, {'x':columnToAnalyse,'y':analysisBasedOnColumn}, 'Data Frame', 'scatter')
                solvingTimeByGroupId = {}
                initialType = None
                for sample in dataFrame[[columnToAnalyse, analysisBasedOnColumn]].values:
                    if not initialType:
                        initialType = type(sample[0])
                    if str(sample[0]) not in solvingTimeByGroupId:
                        solvingTimeByGroupId[str(sample[0])] = [float(sample[-1])]
                    else:
                        solvingTimeByGroupId[str(sample[0])].append(float(sample[-1]))
                    # if sample[0] not in solvingTimeByGroupId:
                    #     solvingTimeByGroupId[sample[0]] = [float(sample[-1])]
                    # else:
                    #     solvingTimeByGroupId[sample[0]].append(float(sample[-1]))

                for k,v in {**solvingTimeByGroupId}.items():
                    solvingTimeByGroupId[k] = sum(v) / len(v)
                dataFrameMean = pd.DataFrame(
                    {
                        columnToAnalyse:[initialType(k) for k in solvingTimeByGroupId.keys()],
                        analysisBasedOnColumn:[initialType(v) for v in solvingTimeByGroupId.values()]
                    },
                    columns=[columnToAnalyse,analysisBasedOnColumn]
                )
                sortedDataFrameMean = dataFrameMean.sort_values(by=[analysisBasedOnColumn])
                log.prettyPython(dataAnalysis, 'sortedDataFrameMean', '\n'+str(sortedDataFrameMean), logLevel=log.STATUS)
                plot(dataFrame, {'x':columnToAnalyse,'y':analysisBasedOnColumn}, 'Data Frame', 'scatter')
                plot(sortedDataFrameMean, {'x':analysisBasedOnColumn,'y':columnToAnalyse}, 'Sorted Data Frame Mean', 'scatter')
                # showData(sortedDataFrameMean, 'no title')
            except Exception as e:
                log.failure(dataAnalysis, f'Not possible to analyse data of {columnToAnalyse} vs {analysisBasedOnColumn}', exception=e)
    input('Hit enter to finish excecussion')


def evaluatePredictions(expected: pd.DataFrame, predicted: pd.DataFrame, muteLogs: bool = False):
    r2 = r2_score(expected, predicted)
    rmse = math.sqrt(mean_squared_error(expected, predicted))
    if not muteLogs:
        log.status(evaluatePredictions, f'SSE: {str(np.sum((expected - predicted)**2)).replace(c.NEW_LINE, c.SPACE_DASH_SPACE)}')
        log.status(evaluatePredictions, f'MSE: {mean_squared_error(expected, predicted)}')
        log.status(evaluatePredictions, f'MAE: {mean_absolute_error(expected, predicted)}')
        log.status(evaluatePredictions, f'RMSE: {rmse}')
        log.status(evaluatePredictions, f'R2-score: {r2}')
    return r2, rmse


def sigmoid(x):
    if abs(x) > 709:
        return 1 / (1 + math.exp(-(709 * (abs(x)/x))))
    return 1 / (1 + math.exp(-x))


def mergeLists(listA: list, listB: list):
    return listA + listB


def mergeMultipleLists(multipleLists: list):
    return reduce(mergeLists, multipleLists)


def getRawDataHandled(rawData, relevantColumns, normalizeIt=False, sigmoidIt=False, statsMap=None):
    rawDataFrame = pd.DataFrame(
        rawData, columns=[*relevantColumns.keys()]
    )
    rawDataFrame = rawDataFrame[
        rawDataFrame[
            [*relevantColumns.keys()]
        ].notnull().all(1)
    ]
    # df["id"] = df['id'].str.replace(',', '').astype(float)
    for columnName in [Constants.OPEN_TIME]:
        if columnName in rawDataFrame.keys():
            try:
                rawDataFrame[columnName] = rawDataFrame[columnName].str.replace(',', '.')
            except Exception as e:
                log.warning(getRawDataHandled, f'Not possible to filter {columnName}', e)
    rawDataFrame = rawDataFrame.astype(
        {k:v for k,v in relevantColumns.items() if v}#, errors='ignore'
    )
    try:
        for columnName in [Constants.CREATED_AT]:
            if columnName in [*rawDataFrame.keys()]:
                # rawDataFrame[columnName] = pd.to_datetime(rawDataFrame[columnName]).values.astype(np.int64) // 10 ** 9
                rawDataFrame[columnName] = pd.to_timedelta(rawDataFrame[columnName].str.split().str[-1]).dt.total_seconds().astype(int)
    except Exception as e:
        log.warning(getRawDataHandled, f'Not possible to handle timestamp {columnName}', e)
    if not statsMap:
        statsMap = {
            Constants.MAX: {},
            Constants.MEAN: {}
        }
        for columnName in relevantColumns:
            if columnName not in [Constants.ID, Constants.SOLVING_TIME]:
                try:
                    statsMap[Constants.MAX][columnName] = 1.0 * rawDataFrame[columnName].max()
                    statsMap[Constants.MEAN][columnName] = statistics.mean(rawDataFrame[columnName])
                except Exception as e:
                    log.warning(getRawDataHandled, f'Not possible to properly populate max and mean of {columnName}', e)
    if normalizeIt:
        for columnName in relevantColumns:
            if columnName not in [Constants.ID, Constants.SOLVING_TIME]:
                if sigmoidIt:
                    rawDataFrame[columnName] = (rawDataFrame[columnName] - statsMap[Constants.MEAN][columnName]) / statsMap[Constants.MAX][columnName]
                    rawDataFrame[columnName] = rawDataFrame[columnName].apply(sigmoid)
                else:
                    rawDataFrame[columnName] = rawDataFrame[columnName] / statsMap[Constants.MAX][columnName]
    elif sigmoidIt:
        for columnName in relevantColumns:
            if columnName not in [Constants.ID, Constants.SOLVING_TIME]:
                rawDataFrame[columnName] = rawDataFrame[columnName].apply(sigmoid)
    return rawDataFrame, statsMap


def replaceData(dataFrame: pd.DataFrame, columnName: str, oldValues: list, newValues: list):
    dataFrameCopy = dataFrame.copy(deep=True)
    # print(columnName, oldValues)
    dataFrameCopy[columnName] = dataFrameCopy[columnName].replace(oldValues, newValues)
    return dataFrameCopy


def splitData(trainningDataFrame: pd.DataFrame, inputColumnNames: list, outputColumnNames: list, testSize: float = 0.2, randomState=None):
    if randomState:
        trainningInputData, testingInputData, trainningOutputData, testingOutputData = train_test_split(
            trainningDataFrame[inputColumnNames],
            trainningDataFrame[[*outputColumn.keys()]],
            test_size=testSize,
            random_state=randomState
        )
    else:
        trainningInputData, testingInputData, trainningOutputData, testingOutputData = train_test_split(
            trainningDataFrame[inputColumnNames],
            trainningDataFrame[outputColumnNames],
            test_size=testSize
        )
    return trainningInputData, testingInputData, trainningOutputData, testingOutputData


def getItMerged(data):
    return pd.DataFrame(
        {columnName:[*columnValues] for columnName, columnValues in data.items()}
    )
