from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd

from python_helper import log, ObjectHelper

import Constants, DataUtils


def buildModel(trainningInputData: list, trainningOutputData: list):
    # print(f'trainningInputData: {type(trainningInputData)}, {trainningInputData}')
    # print(f'trainningInputData[0]: {type(trainningInputData[0])}, {trainningInputData[0]}')
    # print(f'trainningInputData[0][0]: {type(trainningInputData[0][0])}, {trainningInputData[0][0]}')
    # print(f'trainningOutputData: {type(trainningOutputData)}, {trainningOutputData}')
    # print(f'trainningOutputData[0]: {type(trainningOutputData[0])}, {trainningOutputData[0]}')
    # print(f'trainningOutputData[0][0]: {type(trainningOutputData[0][0])}, {trainningOutputData[0][0]}')
    model = linear_model.LinearRegression()
    model.fit(trainningInputData, trainningOutputData)
    return model


def getModel(trainningDataFrame: pd.DataFrame, inputColumns: dict, outputColumn: dict, randomState: bool = None):
    maxR2RandomState = 0
    maxR2 = 0
    for r2RandomState in range(400):
        trainningInputData, testingInputData, trainningOutputData, testingOutputData = DataUtils.splitData(
            trainningDataFrame,
            [*inputColumns.keys()],
            [*outputColumn.keys()],
            testSize=0.2,
            randomState=r2RandomState
        )
        log.debug(getModel, f'Type: {type(trainningInputData)}')
        log.debug(getModel, f'Random state: {r2RandomState}')

        model =  linear_model.LinearRegression()
        model.fit(trainningInputData, trainningOutputData)
        log.debug(getModel, 'Model: predictions = (m * X) + b')
        log.debug(getModel, f'(c) Intercept: {model.intercept_}')
        log.debug(getModel, f'(m) Slope: {model.coef_}')

        evaluatingPredictions = model.predict(testingInputData)

        r2, rmse = DataUtils.evaluatePredictions(testingOutputData, evaluatingPredictions, muteLogs=True)
        if maxR2 < r2:
            maxR2 = r2
            maxR2RandomState = r2RandomState
    log.info(getModel, f'Best r2RandomState: {maxR2RandomState}')
    log.info(getModel, f'R2 score: {maxR2}')

    if randomState:
        trainningInputData, testingInputData, trainningOutputData, testingOutputData = DataUtils.splitData(
            trainningDataFrame,
            [*inputColumns.keys()],
            [*outputColumn.keys()],
            testSize=0.2,
            randomState=randomState
        )
    else:
        trainningInputData = trainningDataFrame[[*inputColumns.keys()]]
        trainningOutputData = trainningDataFrame[[*outputColumn.keys()]]
    return buildModel(trainningInputData, trainningOutputData)
