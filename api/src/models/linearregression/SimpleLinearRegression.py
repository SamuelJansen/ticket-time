from sklearn import linear_model
from python_helper import log

import Constants, DataUtils


# INPUT_COLUMNS = {
#     Constants.AREA_ID: int
#     ,
#     Constants.GROUP_ID: int
#     ,
#     Constants.OPEN_TIME: float
#     ,
#     Constants.TYPE: int
#     # # ,
#     # # Constants.ID: int
#     ,
#     Constants.UNITY_ID: int
#     ,
#     Constants.CREATED_AT: str
#     # # ,
#     # # Constants.REQUESTER_ID: int
#     # # ,
#     # # Constants.SOLVER_ID: int
#     # # ,
#     # # Constants.OPEN_TIME: float
#     ,
#     Constants.OPEN_AT_THE_TIME: int
# }
INPUT_COLUMNS = {
    Constants.AREA_ID: int
    ,
    Constants.GROUP_ID: int
    ,
    Constants.OPEN_TIME: float
    ,
    Constants.TYPE: int
    # # ,
    # # Constants.ID: int
    ,
    Constants.UNITY_ID: int
    # # ,
    # Constants.CREATED_AT: str
    # # ,
    # # Constants.REQUESTER_ID: int
    # # ,
    # # Constants.SOLVER_ID: int
    ,
    Constants.OPEN_AT_THE_TIME: int
}
OUTPUT_COLUMN = {
    Constants.SOLVING_TIME: float
}

RELEVANT_COLUMNS = {
    **INPUT_COLUMNS,
    **OUTPUT_COLUMN
}


def predict(trainningDataFileName, testDataFileName, filterSolvingTimeLesserThan=None, normalizeIt=False, sigmoidIt=False, evaluate=True):
    log.status(predict, f'{__name__}')
    trainningData = DataUtils.readCsv(trainningDataFileName).dropna(thresh=1)
    if filterSolvingTimeLesserThan:
        trainningData = trainningData[trainningData[Constants.SOLVING_TIME] < filterSolvingTimeLesserThan]
    trainningDataFrame, statsMap = DataUtils.getRawDataHandled(
        trainningData,
        {**INPUT_COLUMNS, **OUTPUT_COLUMN},
        normalizeIt=normalizeIt,
        sigmoidIt=sigmoidIt
    )
    toPredictData = DataUtils.readCsv(testDataFileName)
    toPredictDataFrame, statsMap = DataUtils.getRawDataHandled(
        toPredictData,
        {**{Constants.ID: int}, **INPUT_COLUMNS},
        normalizeIt=normalizeIt,
        sigmoidIt=sigmoidIt,
        statsMap=statsMap
    )

    # toPredict = toPredictDataFrame[[*INPUT_COLUMNS.keys()]].values
    # X = trainningDataFrame[[*INPUT_COLUMNS.keys()]].values
    # Y = trainningDataFrame[[*OUTPUT_COLUMN.keys()]][[*OUTPUT_COLUMN.keys()][0]].values
    # lm = linear_model.LinearRegression()
    # model = lm.fit(X,Y)
    # score = model.score(X, Y)
    # predictions = lm.predict(toPredict)
    # log.success(predict, f'score: {score}')
    # assert len(toPredictDataFrame[Constants.ID].values) == len(predictions), f'is {len(toPredictDataFrame[Constants.ID].values)} == {len(predictions)}?'
    # results = DataUtils.getItMerged({
    #     Constants.ID: [*toPredictDataFrame[Constants.ID].values],
    #     Constants.SOLVING_TIME: [*predictions]
    # })
    # return results

    toPredict = toPredictDataFrame[[*INPUT_COLUMNS.keys()]]
    X = trainningDataFrame[[*INPUT_COLUMNS.keys()]]
    Y = trainningDataFrame[[*OUTPUT_COLUMN.keys()]]
    lm = linear_model.LinearRegression()
    model = lm.fit(X,Y)
    predictions = model.predict(toPredict)

    if Constants.SOLVING_TIME in [*toPredictDataFrame.keys()]:
        try:
            DataUtils.evaluatePredictions(toPredictDataFrame[[Constants.SOLVING_TIME]], predictions)
        except Exception as e:
            log.info(predict, 'Not possible to evaluate model over testing data')
            log.warning(predict, f'toPredictDataFrame: {toPredictDataFrame}', exception=e)
    else:
        evaluatingPredictions = model.predict(trainningDataFrame[[*INPUT_COLUMNS.keys()]])
        DataUtils.evaluatePredictions(trainningDataFrame[[Constants.SOLVING_TIME]], evaluatingPredictions)

    assert len(toPredictDataFrame[Constants.ID].values) == len(predictions), f'is {len(toPredictDataFrame[Constants.ID].values)} == {len(predictions)}?'
    results = DataUtils.getItMerged({
        Constants.ID: [*toPredictDataFrame[Constants.ID].values],
        Constants.SOLVING_TIME: [p[0] for p in predictions]
    })
    return results
