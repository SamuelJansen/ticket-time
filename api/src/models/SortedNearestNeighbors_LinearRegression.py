from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import pandas as pd
import numpy as np

from python_helper import log, ObjectHelper

import Constants, DataUtils, NearestNeighborsQuery, LinearRegressionModel, SortedCategoryData, SortData


NEAREST_NEIGHBOR_INPUT_COLUMNS = {
    Constants.AREA_ID: int
    ,
    Constants.GROUP_ID: int
    # ,
    # Constants.OPEN_TIME: float
    ,
    Constants.TYPE: int
    ,
    # Constants.ID: int
    # ,
    Constants.UNITY_ID: int
    # ,
    # Constants.CREATED_AT: str
    # ,
    # Constants.REQUESTER_ID: int
    ,
    Constants.SOLVER_ID: int
    # ,
    # Constants.OPEN_AT_THE_TIME: int
}
LINEAR_REGRESSION_INPUT_COLUMNS = {
    Constants.AREA_ID: int
    ,
    Constants.GROUP_ID: int
    ,
    Constants.OPEN_TIME: float
    ,
    Constants.TYPE: int
    ,
    Constants.ID: int
    ,
    Constants.UNITY_ID: int
    ,
    Constants.CREATED_AT: str
    # ,
    # Constants.REQUESTER_ID: int
    ,
    Constants.SOLVER_ID: int
    ,
    Constants.OPEN_AT_THE_TIME: int
}
OUTPUT_COLUMN = {
    Constants.SOLVING_TIME: float
}


def getPredictions(
    nearestNeighborToPredict,
    linearRegressionToPredict,
    sortedTrainningDataFrame,
    trainningDataFrame,
    nearestNeighborColumns,
    linearRegressionColumns,
    targetDataColumn,
    neighbors
):#(toPredict, trainningData, trainningDataFrame, neighbors):
    predictions = []
    for targetIndex, target in enumerate(nearestNeighborToPredict):
        if targetIndex % 100 == 0:
            log.status(getPredictions, f'Currently predicting the {targetIndex}° test case')
        nearestNeighborsTrainningDataSubset, trainningDataSubsetIndexList = NearestNeighborsQuery.getBestFitList(target, sortedTrainningDataFrame[nearestNeighborColumns].values, [*sortedTrainningDataFrame.values], neighbors)

        linearRegressionTrainningData = trainningDataFrame[{**linearRegressionColumns, **targetDataColumn}].values
        linearRegressionTrainningDataSubset = [linearRegressionTrainningData[i] for i in trainningDataSubsetIndexList]

        innerInputDataSubset = np.asarray([np.asarray([*v[:-1]]) for v in linearRegressionTrainningDataSubset])
        innerOutputDataSubset = np.asarray([[v[-1]] for v in linearRegressionTrainningDataSubset])

        # print(innerInputDataSubset)
        # print(innerOutputDataSubset)
        # print(linearRegressionToPredict[targetIndex].reshape(1, -1))

        model = LinearRegressionModel.buildModel(innerInputDataSubset, innerOutputDataSubset)
        prediction = model.predict(linearRegressionToPredict[targetIndex].reshape(1, -1))
        # print(f'prediction: {type(prediction)}, {prediction}')
        # input('test')

        predictions.append(prediction[0])

    predictions = np.asarray(predictions)
    return predictions


def getSortedPredictions(
    trainningDataFrame,
    toPredictDataFrame,
    neighbors,
    statsMap,
    nearestNeighborColumns,
    linearRegressionColumns,
    targetDataColumn,
    relevantInputColumns,
    relevantOutputColumns
):
    sortedCategoryData = SortedCategoryData.SortedCategoryData()
    sortedTrainningDataFrame = sortedCategoryData.sortData(
        trainningDataFrame,
        [*relevantInputColumns.keys()],
        [*relevantOutputColumns.keys()][0],
        statsMap
    )
    sortedToPredict = SortData.sortIt(
        toPredictDataFrame[[*relevantInputColumns.keys()]],
        [*relevantInputColumns.keys()],
        sortedCategoryData.categoriesMap
    )

    nearestNeighborToPredict = sortedToPredict[[*nearestNeighborColumns.keys()]].values
    linearRegressionToPredict = toPredictDataFrame[[*linearRegressionColumns.keys()]].values

    predictions = getPredictions(
        nearestNeighborToPredict,
        linearRegressionToPredict,
        sortedTrainningDataFrame,
        trainningDataFrame,
        nearestNeighborColumns,
        linearRegressionColumns,
        targetDataColumn,
        neighbors
    )

    minOutput = min(trainningDataFrame[[targetDataColumn.keys()][0]].values)
    maxOutput = max(trainningDataFrame[[targetDataColumn.keys()][0]].values)
    for p in predictions:
        if p[0] < minOutput:
            # print(p[0])
            p[0] = minOutput * (DataUtils.sigmoid(p[0] - minOutput) + 0.5)
        if p[0] > maxOutput:
            p[0] = maxOutput + maxOutput * (DataUtils.sigmoid(p[0] - maxOutput) - 0.5)

    return predictions


def predict(
    trainningDataFileName: str,
    testDataFileName: str,
    neighbors: int = 5,
    normalizeIt: bool = False,
    sigmoidIt: bool = False,
    evaluate: bool = False,
    nearestNeighborColumns: dict = NEAREST_NEIGHBOR_INPUT_COLUMNS,
    linearRegressionColumns: dict = LINEAR_REGRESSION_INPUT_COLUMNS,
    targetDataColumn: dict = OUTPUT_COLUMN,
    filterSolvingTimeLesserThan: float = None
):
    log.status(predict, f'{__name__}')
    trainningData = DataUtils.readCsv(trainningDataFileName).dropna(thresh=1)
    if filterSolvingTimeLesserThan:
        trainningData = trainningData[trainningData[targetDataColumn.keys()[0]] < filterSolvingTimeLesserThan]
    # print(trainningData)

    relevantInputColumns: dict = {**nearestNeighborColumns, **linearRegressionColumns}
    relevantOutputColumns: dict = {**targetDataColumn}
    relevantColumns: dict = {**relevantInputColumns, **relevantOutputColumns}

    trainningDataFrame, statsMap = DataUtils.getRawDataHandled(
        trainningData,
        relevantColumns,
        normalizeIt=normalizeIt,
        sigmoidIt=sigmoidIt
    )
    # print(trainningDataFrame)

    toPredictData = DataUtils.readCsv(testDataFileName)
    # print('toPredictData[ID].values', toPredictData[ID].values)

    toPredictDataFrameException = None
    try:
        toPredictDataFrame, _ = DataUtils.getRawDataHandled(
            toPredictData,
            {**{Constants.ID: int}, **relevantInputColumns} if not evaluate else {
                **{Constants.ID: int},
                **relevantInputColumns,
                **relevantOutputColumns
            },
            normalizeIt=normalizeIt,
            sigmoidIt=sigmoidIt,
            statsMap=statsMap
        )
        assert 0 < len(toPredictDataFrame.values), toPredictDataFrame
    except Exception as e:
        toPredictDataFrameException = e
        log.log(predict, 'Not possible to include target column', exception=toPredictDataFrameException)
    if ObjectHelper.isNotNone(toPredictDataFrameException) or 0 == len([*toPredictDataFrame.values]) :
        toPredictDataFrame, _ = DataUtils.getRawDataHandled(
            toPredictData,
            {**{Constants.ID: int}, **relevantInputColumns},
            normalizeIt=normalizeIt,
            sigmoidIt=sigmoidIt,
            statsMap=statsMap
        )
    assert 0 < len(toPredictDataFrame.values), f'toPredictDataFrame: {toPredictDataFrame}'

    # pedictions = getSortedPredictions(
    #     trainningDataFrame,
    #     toPredictDataFrame,
    #     neighbors,
    #     statsMap,
    #     nearestNeighborColumns,
    #     linearRegressionColumns,
    #     targetDataColumn,
    #     relevantInputColumns,
    #     relevantOutputColumns
    # )

    # print(trainningDataFrame)
    # print(toPredictDataFrame)
    # print(predictions[:10])

    if [*targetDataColumn.keys()][0] in [*toPredictDataFrame.keys()]:
        try:
            DataUtils.evaluatePredictions(toPredictDataFrame[[[*targetDataColumn.keys()][0]]], predictions)
        except Exception as e:
            log.info(predict, 'Not possible to evaluate model over testing data')
            log.warning(predict, f'toPredictDataFrame: {toPredictDataFrame}', exception=e)
    else:
        ITERATIONS = 10
        TESTS = 6
        NEIGHBORS_START = 5
        NEIGHBORS_PASS = 20
        RMSE_CAP = 1000.0
        testingStatsMap = {}
        for iteration in range(ITERATIONS):
            testNeighbors = iteration * NEIGHBORS_PASS + NEIGHBORS_START
            testingStatsMap[testNeighbors] = {
                Constants.R2 : 0.0,
                Constants.RMSE : 0.0
            }
            for n in range(TESTS):
                test_trainningInputData, test_testingInputData, test_trainningOutputData, test_testingOutputData = DataUtils.splitData(
                    trainningDataFrame,
                    [*relevantInputColumns.keys()],
                    [*relevantOutputColumns.keys()],
                    testSize=0.2
                )
                test_trainningDataFrame = pd.merge(test_trainningInputData, test_trainningOutputData, left_index=True, right_index=True)
                test_toPredictDataFrame = pd.merge(test_testingInputData, test_testingOutputData, left_index=True, right_index=True)
                evaluatingPredictions = getSortedPredictions(
                    test_trainningDataFrame,
                    test_toPredictDataFrame,
                    neighbors,
                    statsMap,
                    nearestNeighborColumns,
                    linearRegressionColumns,
                    targetDataColumn,
                    relevantInputColumns,
                    relevantOutputColumns
                )
                r2, rmse = DataUtils.evaluatePredictions(
                    np.asarray([[ep[0]] for ep in test_testingOutputData.values]),
                    np.asarray([[ep[0]] for ep in evaluatingPredictions])
                )
                testingStatsMap[testNeighbors][Constants.R2] += r2
                testingStatsMap[testNeighbors][Constants.RMSE] += (rmse / RMSE_CAP)
            testingStatsMap[testNeighbors][Constants.R2] = testingStatsMap[testNeighbors][Constants.R2] / TESTS
            testingStatsMap[testNeighbors][Constants.RMSE] = (testingStatsMap[testNeighbors][Constants.RMSE] / TESTS) * RMSE_CAP
            log.prettyPython(predict, f'Iteration: {iteration} testNeighbors: {testNeighbors}, Stats map', testingStatsMap, logLevel=log.STATUS)
        log.prettyPython(predict, 'Stats map completed', testingStatsMap, logLevel=log.STATUS)

    assert len(toPredictDataFrame[Constants.ID].values) == len(predictions), f'is {len(toPredictDataFrame[Constants.ID].values)} == {len(predictions)}?'
    results = DataUtils.getItMerged({
        Constants.ID: [*toPredictDataFrame[Constants.ID].values],
        [*targetDataColumn.keys()][0]: [p[0] for p in predictions]
    })
    return results
