from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import pandas as pd
import numpy as np

from python_helper import log, ObjectHelper

import Constants, DataUtils, SortedCategoryData, SortData, LinearRegressionModel


INPUT_COLUMNS = {
    Constants.AREA_ID: int
    ,
    Constants.GROUP_ID: int
    # ,
    # Constants.OPEN_TIME: float
    ,
    Constants.TYPE: int
    # ,
    # Constants.ID: int
    ,
    Constants.UNITY_ID: int
    # ,
    # Constants.CREATED_AT: str
    # # ,
    # # Constants.REQUESTER_ID: int
    # ,
    # Constants.SOLVER_ID: int
    # ,
    # Constants.OPEN_AT_THE_TIME: int
}
OUTPUT_COLUMN = {
    Constants.SOLVING_TIME: float
}

RELEVANT_COLUMNS = {
    **INPUT_COLUMNS,
    **OUTPUT_COLUMN
}


def predict(trainningDataFileName, testDataFileName, evaluate=True, filterSolvingTimeLesserThan=None, normalizeIt=False, sigmoidIt=False, randomState=None):
    log.status(predict, f'{__name__}')
    trainningData = DataUtils.readCsv(trainningDataFileName).dropna(thresh=1)
    if filterSolvingTimeLesserThan:
        trainningData = trainningData[trainningData[Constants.SOLVING_TIME] < filterSolvingTimeLesserThan]
    # print(trainningData)

    trainningDataFrame, statsMap = DataUtils.getRawDataHandled(
        trainningData,
        {**INPUT_COLUMNS, **OUTPUT_COLUMN},
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
            {**{Constants.ID: int}, **INPUT_COLUMNS} if not evaluate else {
                **{Constants.ID: int},
                **INPUT_COLUMNS,
                **OUTPUT_COLUMN
            },
            normalizeIt=normalizeIt,
            sigmoidIt=sigmoidIt,
            statsMap=statsMap
        )
    except Exception as e:
        toPredictDataFrameException = e
        log.warning(predict, 'Not possible to include target column', exception=toPredictDataFrameException)
    if ObjectHelper.isNotNone(toPredictDataFrameException) or 0 == len([*toPredictDataFrame.values]) :
        toPredictDataFrame, _ = DataUtils.getRawDataHandled(
            toPredictData,
            {**{Constants.ID: int}, **INPUT_COLUMNS},
            normalizeIt=normalizeIt,
            sigmoidIt=sigmoidIt,
            statsMap=statsMap
        )
    assert 0 < len(toPredictDataFrame.values), toPredictDataFrame

    toPredict = toPredictDataFrame[[*INPUT_COLUMNS.keys()]]
    # print(toPredict)

    sortedCategoryData = SortedCategoryData.SortedCategoryData()
    sortedTrainningDataFrame = sortedCategoryData.sortData(trainningDataFrame, [*INPUT_COLUMNS.keys()], [*OUTPUT_COLUMN.keys()][0], statsMap)
    sortedToPredict = SortData.sortIt(toPredict, [*INPUT_COLUMNS.keys()], sortedCategoryData.categoriesMap)

    model = LinearRegressionModel.getModel(sortedTrainningDataFrame, INPUT_COLUMNS, OUTPUT_COLUMN, randomState=randomState)
    predictions = model.predict(sortedToPredict)

    # print(sortedTrainningDataFrame)
    # print(sortedToPredict)
    # print(predictions[:10])
    minOutput = min(trainningDataFrame[[OUTPUT_COLUMN.keys()][0]].values)
    maxOutput = max(trainningDataFrame[[OUTPUT_COLUMN.keys()][0]].values)
    for p in predictions:
        if p[0] < minOutput:
            # print(p[0])
            p[0] = minOutput
        if p[0] > maxOutput:
            p[0] = maxOutput


    if evaluate:
        if Constants.SOLVING_TIME in [*toPredictDataFrame.keys()]:
            try:
                # print(toPredictDataFrame[[Constants.SOLVING_TIME]])
                # print(predictions)
                DataUtils.evaluatePredictions(toPredictDataFrame[[Constants.SOLVING_TIME]], predictions)
            except Exception as e:
                log.info(predict, 'Not possible to evaluate model over testing data')
                log.warning(predict, f'toPredictDataFrame: {toPredictDataFrame}', exception=e)
        else:
            evaluatingPredictions = model.predict(sortedTrainningDataFrame[[*INPUT_COLUMNS.keys()]])
            DataUtils.evaluatePredictions(sortedTrainningDataFrame[[Constants.SOLVING_TIME]], evaluatingPredictions)

    assert len(toPredictDataFrame[Constants.ID].values) == len(predictions), f'is {len(toPredictDataFrame[Constants.ID].values)} == {len(predictions)}?'
    results = DataUtils.getItMerged({
        Constants.ID: [*toPredictDataFrame[Constants.ID].values],
        Constants.SOLVING_TIME: [p[0] for p in predictions]
    })
    return results
