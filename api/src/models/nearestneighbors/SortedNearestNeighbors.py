import numpy as np

from python_helper import log, ObjectHelper

import Constants, DataUtils, SortedCategoryData, SortData, NearestNeighborsQuery


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


def predict(trainningDataFileName, testDataFileName, neighbors=5, evaluate=False, filterSolvingTimeLesserThan=None, normalizeIt=False, sigmoidIt=False):
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
        assert 0 < len(toPredictDataFrame.values), toPredictDataFrame
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

    sortedCategoryData = SortedCategoryData.SortedCategoryData()

    sortedTrainningDataFrame = sortedCategoryData.sortData(
        trainningDataFrame,
        [*INPUT_COLUMNS.keys()],
        [*OUTPUT_COLUMN.keys()][0],
        statsMap
    )

    toPredict = SortData.sortIt(
        toPredictDataFrame[[*INPUT_COLUMNS.keys()]],
        [*INPUT_COLUMNS.keys()],
        sortedCategoryData.categoriesMap
    ).values
    # print(toPredict)

    X = (sortedTrainningDataFrame[[*INPUT_COLUMNS.keys()]]).values
    # print('X', X)
    Y = (sortedTrainningDataFrame[[*OUTPUT_COLUMN.keys()][-1]]).values
    # print('Y', Y)

    predictions = NearestNeighborsQuery.getPredictions(toPredict, X, sortedTrainningDataFrame, neighbors)

    # print(sortedTrainningDataFrame)
    # print(SortData.sortIt(
    #     toPredictDataFrame[[*INPUT_COLUMNS.keys()]],
    #     [*INPUT_COLUMNS.keys()],
    #     sortedCategoryData.categoriesMap
    # ))
    # print(predictions[:10])

    SLICE_SIZE = 10
    if evaluate:
        if Constants.SOLVING_TIME in toPredictDataFrame.keys():
            try:
                DataUtils.evaluatePredictions(predictions, toPredictDataFrame[Constants.SOLVING_TIME])
            except Exception as e:
                log.warning(predict, f'Not possible to evaluate model on actual testing data', exception=e)
        else:
            evaluatingPredictions = NearestNeighborsQuery.getPredictions(X, X, sortedTrainningDataFrame, neighbors)
            log.status(predict, f'evaluatingPredictions: {evaluatingPredictions[:10]}')
            log.status(predict, f'Y: {Y[:10]}')
            DataUtils.evaluatePredictions(Y, evaluatingPredictions)

    assert len(toPredictDataFrame[Constants.ID].values) == len(predictions), f'is {len(toPredictDataFrame[Constants.ID].values)} == {len(predictions)}?'
    results = DataUtils.getItMerged({
        Constants.ID: [*toPredictDataFrame[Constants.ID].values],
        Constants.SOLVING_TIME: [*predictions]
    })
    if evaluate:
        log.status(predict, f'results: {results[:SLICE_SIZE]}')
    return results
