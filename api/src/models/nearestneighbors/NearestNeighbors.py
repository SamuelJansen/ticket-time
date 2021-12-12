import numpy as np

from python_helper import log, ObjectHelper

import Constants, DataUtils, NearestNeighborsQuery


def predict(trainningDataFileName, testDataFileName, neighbors=5, evaluate=False, filterSolvingTimeLesserThan=None, normalizeIt=False, sigmoidIt=False):
    log.status(predict, f'{__name__}')
    trainningData = DataUtils.readCsv(trainningDataFileName).dropna(thresh=1)
    if filterSolvingTimeLesserThan:
        trainningData = trainningData[trainningData[Constants.SOLVING_TIME] < filterSolvingTimeLesserThan]
    # print(trainningData)

    trainningDataFrame, statsMap = DataUtils.getRawDataHandled(
        trainningData,
        {**Constants.INPUT_COLUMNS, **Constants.OUTPUT_COLUMN},
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
            {**{Constants.ID: int}, **Constants.INPUT_COLUMNS} if not evaluate else {
                **{Constants.ID: int},
                **Constants.INPUT_COLUMNS,
                **Constants.OUTPUT_COLUMN
            },
            normalizeIt=normalizeIt,
            sigmoidIt=sigmoidIt,
            statsMap=statsMap
        )
        assert 0 < len(toPredictDataFrame.values), toPredictDataFrame
    except Exception as e:
        toPredictDataFrameException = e
        log.warning(predict, 'Not possible to include target column', exception=toPredictDataFrameException)
    if ObjectHelper.isNotNone(toPredictDataFrameException) or 0 == len([*toPredictDataFrame.values]) :
        toPredictDataFrame, _ = DataUtils.getRawDataHandled(
            toPredictData,
            {**{Constants.ID: int}, **Constants.INPUT_COLUMNS},
            normalizeIt=normalizeIt,
            sigmoidIt=sigmoidIt,
            statsMap=statsMap
        )
    assert 0 < len(toPredictDataFrame.values), toPredictDataFrame

    toPredict = toPredictDataFrame[[*Constants.INPUT_COLUMNS.keys()]].values
    # print(toPredict)

    X = trainningDataFrame[[*Constants.INPUT_COLUMNS.keys()]].values
    # print('X', X)
    Y = trainningDataFrame[[*Constants.OUTPUT_COLUMN.keys()][-1]].values
    # print('Y', Y)

    predictions = NearestNeighborsQuery.getPredictions(toPredict, X, trainningDataFrame, neighbors)

    SLICE_SIZE = 10
    if evaluate:
        if Constants.SOLVING_TIME in toPredictDataFrame.keys():
            try:
                DataUtils.evaluatePredictions(predictions, toPredictDataFrame[Constants.SOLVING_TIME])
            except Exception as e:
                log.warning(predict, f'Not possible to evaluate model on actual testing data', exception=e)
        else:
            evaluatingPredictions = NearestNeighborsQuery.getPredictions(X, X, trainningDataFrame, neighbors)
            log.status(predict, f'evaluatingPredictions: {evaluatingPredictions}')
            log.status(predict, f'Y: {Y}')
            DataUtils.evaluatePredictions(Y, evaluatingPredictions)

    assert len(toPredictDataFrame[Constants.ID].values) == len(predictions), f'is {len(toPredictDataFrame[Constants.ID].values)} == {len(predictions)}?'
    results = DataUtils.getItMerged({
        Constants.ID: [*toPredictDataFrame[Constants.ID].values],
        Constants.SOLVING_TIME: [*predictions]
    })
    if evaluate:
        log.status(predict, f'results: {results[:SLICE_SIZE]}')
    return results
