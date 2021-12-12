import datetime
import pandas as pd
from sklearn import linear_model
import numpy as np
from functools import reduce
from python_helper import EnvironmentHelper, StringHelper, log
import globals


globalsInstance = globals.newGlobalsInstance(
        __file__
        , successStatus = True
        , errorStatus = True
        , failureStatus = True
        # , settingStatus = True
        , statusStatus = True
        , infoStatus = True
        # , debugStatus = True
        , warningStatus = True
        # , wrapperStatus = True
        # , testStatus = True
        # , logStatus = True
    )


import Constants, DataUtils
import SimpleLinearRegression, LinearRegression, SortedLinearRegression
import NearestNeighbors, SortedNearestNeighbors
import NearestNeighbors_LinearRegression, SortedNearestNeighbors_LinearRegression
import CO2EmissionExample


# DataUtils.dataAnalysis('train.csv', Constants.INPUT_COLUMNS, analysisBasedOnColumn=Constants.SOLVING_TIME, skipPlots=True)
# DataUtils.dataAnalysis('test.csv', Constants.INPUT_COLUMNS, skipPlots=True)

# co2EmissionExample = CO2EmissionExample.predict('FuelConsumptionCo2.csv')
# DataUtils.toCsv(co2EmissionExample, 'co2-emission-linear-regression-results.csv')


# simpleLinearRegressionResults = SimpleLinearRegression.predict('train.csv', 'test.csv', normalizeIt=False, sigmoidIt=False, evaluate=True)
# simpleLinearRegressionResults = SimpleLinearRegression.predict('cleaned_train_data.csv', 'test.csv')
# simpleLinearRegressionResults = SimpleLinearRegression.predict('2_train.csv', '2_train.csv', normalizeIt=False, sigmoidIt=False, evaluate=True)
# simpleLinearRegressionResults = SimpleLinearRegression.predict('3_train.csv', '3_test.csv', normalizeIt=False, sigmoidIt=False, evaluate=True)
# DataUtils.toCsv(simpleLinearRegressionResults, 'simple-linear-regression-results.csv')

# linearRegressionResults = LinearRegression.predict('train.csv', 'test.csv', normalizeIt=False, sigmoidIt=False, evaluate=True)
# linearRegressionResults = LinearRegression.predict('0_train.csv', '0_test.csv')
# linearRegressionResults = LinearRegression.predict('1_train.csv', '1_test.csv')
# linearRegressionResults = LinearRegression.predict('2_train.csv', '2_test.csv', normalizeIt=True, sigmoidIt=True, evaluate=True)
# linearRegressionResults = LinearRegression.predict('3_train.csv', '3_test.csv', normalizeIt=True, sigmoidIt=True, evaluate=True)
# DataUtils.toCsv(linearRegressionResults, 'test-linear-regression-results.csv')


# sortedLinearRegressionResults = SortedLinearRegression.predict('train.csv', 'test.csv', normalizeIt=True, sigmoidIt=True, evaluate=True)
# sortedLinearRegressionResults = SortedLinearRegression.predict('0_train.csv', '0_test.csv')
# sortedLinearRegressionResults = SortedLinearRegression.predict('1_train.csv', '1_test.csv')
# sortedLinearRegressionResults = SortedLinearRegression.predict('2_train.csv', '2_test.csv', normalizeIt=True, sigmoidIt=True, evaluate=True)
# sortedLinearRegressionResults = SortedLinearRegression.predict('3_train.csv', '3_test.csv', normalizeIt=True, sigmoidIt=True, evaluate=True)
# DataUtils.toCsv(sortedLinearRegressionResults, 'test-sorted-linear-regression-results.csv')


# nearesNeighborsResults = NearestNeighbors.predict('train.csv', 'test.csv', neighbors=5, evaluate=True, normalizeIt=True)
# nearesNeighborsResults = NearestNeighbors.predict('train.csv', 'train.csv', neighbors=1, evaluate=True)
# nearesNeighborsResults = NearestNeighbors.predict('0_train.csv', '0_test.csv', neighbors=30, evaluate=True)
# nearesNeighborsResults = NearestNeighbors.predict('1_train.csv', '1_test.csv', neighbors=3, evaluate=True)
# nearesNeighborsResults = NearestNeighbors.predict('2_train.csv', '2_test.csv', neighbors=3, normalizeIt=True, sigmoidIt=True, evaluate=True)
# nearesNeighborsResults = NearestNeighbors.predict('3_train.csv', '3_test.csv', neighbors=3, normalizeIt=True, sigmoidIt=True, evaluate=True)
# DataUtils.toCsv(nearesNeighborsResults, 'nearest-neighbors-results.csv')


# sortedNearestNeighborsResults = SortedNearestNeighbors.predict('train.csv', 'test.csv', neighbors=30, evaluate=True, normalizeIt=True)
# sortedNearestNeighborsResults = SortedNearestNeighbors.predict('train.csv', 'train.csv', neighbors=1, evaluate=True)
# sortedNearestNeighborsResults = SortedNearestNeighbors.predict('0_train.csv', '0_test.csv', neighbors=30, evaluate=True)
# sortedNearestNeighborsResults = SortedNearestNeighbors.predict('1_train.csv', '1_test.csv', neighbors=3, normalizeIt=True, sigmoidIt=True, evaluate=True)
# sortedNearestNeighborsResults = SortedNearestNeighbors.predict('2_train.csv', '2_test.csv', neighbors=3, normalizeIt=True, sigmoidIt=True, evaluate=True)
# DataUtils.toCsv(sortedNearestNeighborsResults, 'sorted-normalized-nearest-neighbors-results.csv')


NN_LR_NEIGHBORS = 750
nn_lr = NearestNeighbors_LinearRegression.predict('train.csv', 'test.csv', neighbors=NN_LR_NEIGHBORS, normalizeIt=True, sigmoidIt=True, evaluate=True)
# nn_lr = NearestNeighbors_LinearRegression.predict('2_train.csv', '2_test.csv', neighbors=NN_LR_NEIGHBORS, normalizeIt=True, sigmoidIt=True, evaluate=True)
# nn_lr = NearestNeighbors_LinearRegression.predict('3_train.csv', '3_test.csv', neighbors=NN_LR_NEIGHBORS, normalizeIt=True, sigmoidIt=True, evaluate=True)
# DataUtils.toCsv(nn_lr, f'from-{NN_LR_NEIGHBORS}-nearest-neighbors-to-linear-regression-results.csv')



# SNN_LR_NEIGHBORS = 750
# snn_lr = SortedNearestNeighbors_LinearRegression.predict('train.csv', 'test.csv', neighbors=SNN_LR_NEIGHBORS, normalizeIt=True, sigmoidIt=True, evaluate=True)
# snn_lr = SortedNearestNeighbors_LinearRegression.predict('2_train.csv', '2_test.csv', neighbors=SNN_LR_NEIGHBORS, normalizeIt=True, sigmoidIt=True, evaluate=True)
# snn_lr = SortedNearestNeighbors_LinearRegression.predict('3_train.csv', '3_test.csv', neighbors=SNN_LR_NEIGHBORS, normalizeIt=True, sigmoidIt=True, evaluate=True)
