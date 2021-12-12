import pandas as pd


import SortData, Constants


DO_NOT_TOUCH_THIS_COLUMNS = [Constants.ID, Constants.SOLVING_TIME]


class SortedCategoryData:

    def sortData(self, data: pd.DataFrame, inputColumns: list, outputColumn: str, statsMap: dict):
        self.resetModel(inputColumns, outputColumn, statsMap)
        self.data: pd.DataFrame = data
        sortedData = self.data.copy()
        for columnName in self.relevantColumns:
            if columnName not in DO_NOT_TOUCH_THIS_COLUMNS:
                sortedData, categories = SortData.getItSorted(sortedData, columnName, self.outputColumn)
                self.categoriesMap[columnName] = {**categories, **self.categoriesMap[columnName]}
        self.sortedData = sortedData
        return self.getSortedData()

    # def predict(self, testingData: pd.DataFrame):
    #     sortedTestingData = sortIt(testingData, self.relevantColumns, self.categoriesMap)
    #     print(f'sortedData: {self.getSortedData()}')
    #     print(f'sortedTestingData: {sortedTestingData}')

    def resetModel(self, inputColumns: list, outputColumn: str, statsMap: dict):
        self.outputColumn = outputColumn
        self.relevantColumns = [*inputColumns, outputColumn]
        self.categoriesMap = {
            k: {
                Constants.MAX: statsMap[Constants.MAX].get(k, 1.0),
                Constants.MEAN: statsMap[Constants.MEAN].get(k, 0.5)
            } for k in self.relevantColumns
        }
        # print(f'self.categoriesMap: {self.categoriesMap}')

    def getSortedData(self):
        return self.sortedData.copy()
