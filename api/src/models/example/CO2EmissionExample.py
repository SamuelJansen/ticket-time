from matplotlib import pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from math import sqrt
from python_helper import EnvironmentHelper, StringHelper, log


import DataUtils


INPUT_COLUMNS = {
    'MODELYEAR': float,
    # 'MAKE': str,
    # 'MODEL': str,
    # 'VEHICLECLASS': str,
    'ENGINESIZE': float,
    'CYLINDERS': float,
    # 'TRANSMISSION': str,
    # 'FUELTYPE': str,
    'FUELCONSUMPTION_CITY': float,
    'FUELCONSUMPTION_HWY': float,
    'FUELCONSUMPTION_COMB': float,
    'FUELCONSUMPTION_COMB_MPG': float
}
OUTPUT_COLUMN = {
    'CO2EMISSIONS': float
}


def predict(csvFileName, inputColumns=INPUT_COLUMNS, outputColumn=OUTPUT_COLUMN):
    df = DataUtils.readCsv(csvFileName)
    log.debug(predict, df.head())
    log.debug(predict, df.describe())

    # engines = df[['ENGINESIZE']]
    engines = df[[columnName for columnName in inputColumns.keys()]]
    co2 = df[[columnName for columnName in outputColumn.keys()]]
    log.debug(predict, engines.head())

    engines_train, engines_test, co2_train, co2_test = train_test_split(engines, co2, test_size=0.2, random_state=42)
    log.debug(predict, type(engines_train))
    log.debug(predict, f'Random state: {check_random_state()}')

    # plt.scatter(engines_train, co2_train[['CO2EMISSIONS']], color='blue')
    # plt.title('Train Data')
    # plt.xlabel('Engine')
    # plt.ylabel('C02 emission')
    # plt.show()

    model =  linear_model.LinearRegression()
    model.fit(engines_train, co2_train)
    log.debug(predict, '(A) Intercept: ', model.intercept_)
    log.debug(predict, '(B) Slope: ', model.coef_)

    # plt.scatter(engines_train, co2_train[['CO2EMISSIONS']], color='blue')
    # plt.plot(engines_train, model.coef_[0][0]*engines_train + model.intercept_[0], '-r')
    # plt.title('Linear Regression over Train Data')
    # plt.ylabel('C02 emission')
    # plt.xlabel('Engines')
    # plt.show()

    predictedCo2Emission = model.predict(engines_test)

    # plt.scatter(engines_test, co2_test[['CO2EMISSIONS']], color='blue')
    # # plt.plot(engines_test, model.coef_[0][0]*engines_test + model.intercept_[0], '-r')
    # plt.title('Linear Regression vs Test Data')
    # plt.ylabel('C02 emission')
    # plt.xlabel('Engines')
    # plt.show()

    log.status(predict, f'SSE: {np.sum((predictedCo2Emission - co2_test)**2)}')
    log.status(predict, f'MSE: {mean_squared_error(co2_test, predictedCo2Emission)}')
    log.status(predict, f'MAE: {mean_absolute_error(co2_test, predictedCo2Emission)}')
    log.status(predict, f'RMSE: {sqrt(mean_squared_error(co2_test, predictedCo2Emission))}')
    log.status(predict, f'R2-score: {r2_score(co2_test, predictedCo2Emission)}')

    # input('Hit enter to finish excecussion')
    return predictedCo2Emission
