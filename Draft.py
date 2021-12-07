import datetime
import pandas as pd
from sklearn import linear_model
import numpy as np
from functools import reduce
from python_helper import EnvironmentHelper, StringHelper, log
import globals


ID = 'ID_TICKET'
TYPE = 'TIPOTICKET'
AREA_ID = 'ID_AREA'
GROUP_ID = 'ID_GRUPO_SERVICO'
UNITY_ID = 'ID_UNIDADE'
SOLVING_TIME = 'TEMPOMINUTO'
OPEN_TIME = 'MIN_ABERTO'
CREATED_AT = 'DT_CRIACAO'
OPEN_AT_THE_TIME = 'QTD_TOTAL_ABERTOS_GRUPO_SERV'

TRAINNING_COLUMNS = {
    AREA_ID: int
    ,
    GROUP_ID: int
    ,
    OPEN_TIME: float
    ,
    TYPE: int
    ,
    ID: int
    # ,
    # UNITY_ID: int
    # ,
    # CREATED_AT: str
    # ,
    # 'IDQUEMABRIUTICKET': int
    # ,
    # 'IDTECRESP': int
    # ,
    # OPEN_AT_THE_TIME: int
}
TARGET_COLUMN = {
    SOLVING_TIME: float
}

RELEVANT_COLUMNS = {
    **TRAINNING_COLUMNS,
    **TARGET_COLUMN
}


globalsInstance = globals.newGlobalsInstance(
        __file__
        , successStatus = True
        , errorStatus = True
        , failureStatus = True
        # , settingStatus = True
        , statusStatus = True
        , infoStatus = True
        , debugStatus = True
        , warningStatus = True
        # , wrapperStatus = True
        # , testStatus = True
        # , logStatus = True
    )

def getFileNameWithPath(fileName):
    return StringHelper.join(['api', 'resource', fileName], EnvironmentHelper.OS_SEPARATOR)


def readCsv(csvFileName):
    return pd.read_csv(getFileNameWithPath(csvFileName))


def toCsv(content, csvFileName):
    # compression_opts = dict(method='zip', archive_name=getFileNameWithPath(csvFileName))
    # content.to_csv(csvFileName, index=False, compression=compression_opts)
    content.to_csv(getFileNameWithPath(csvFileName), index=False)


def getRawDataHandled(trainningData, relevantColumns):
    trainningDataFrame = pd.DataFrame(
        trainningData, columns=[*relevantColumns.keys()]
    )
    trainningDataFrame = trainningDataFrame[
        trainningDataFrame[
            [*relevantColumns.keys()]
        ].notnull().all(1)
    ]
    # df["id"] = df['id'].str.replace(',', '').astype(float)
    for columnName in [OPEN_TIME]:
        try:
            trainningDataFrame[columnName] = trainningDataFrame[columnName].str.replace(',', '.')
        except Exception as e:
            log.warning(log.warning, f'Not possible to filter {columnName}', e)
    trainningDataFrame = trainningDataFrame.astype(
        relevantColumns#, errors='ignore'
    )
    try:
        for columnName in [CREATED_AT]:
            trainningDataFrame[columnName] = pd.to_datetime(trainningDataFrame[columnName]).values.astype(np.int64) // 10 ** 9
    except Exception as e:
        log.warning(log.warning, f'Not possible to handle timestamp {columnName}', e)
    return trainningDataFrame


def getItMerged(data):
    return pd.DataFrame(
        {columnName:[*columnValues] for columnName, columnValues in data.items()}
    )


def linearRegressionStrategy():
    trainningData = readCsv('train.csv').dropna(thresh=1)
    # print(trainningData)

    trainningDataFrame = getRawDataHandled(trainningData, {**TRAINNING_COLUMNS, **TARGET_COLUMN})
    # print(trainningDataFrame)

    toPredictData = readCsv('test.csv')
    # print('toPredictData[ID].values', toPredictData[ID].values)

    toPredictDataFrame = getRawDataHandled(toPredictData, TRAINNING_COLUMNS)
    # print(trainningDataFrame)
    toPredict = toPredictDataFrame[[*TRAINNING_COLUMNS.keys()]].values
    # print(toPredict)

    X = trainningDataFrame[[*TRAINNING_COLUMNS.keys()]].values
    # print('X', X)
    Y = trainningDataFrame[[*TARGET_COLUMN.keys()]][[*TARGET_COLUMN.keys()][0]].values
    # print('Y', Y)

    lm = linear_model.LinearRegression()
    model = lm.fit(X,Y)
    print(model)

    score = model.score(X, Y)
    predictions = lm.predict(toPredict)
    print('predictions', predictions, 'score', score)

    assert len(toPredictData[ID].values) == len(predictions), f'is {len(toPredictData[ID].values)} == {len(predictions)}?'
    results = getItMerged({
        ID: [*toPredictData[ID].values],
        SOLVING_TIME: [*predictions]
    })
    # print('results', results)

    # toCsv(results, 'linear-regression-results.csv')
linearRegressionStrategy()
