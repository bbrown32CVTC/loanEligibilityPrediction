# This is a Python script for predicting loan eligibility

import numpy as np
import pandas as pd


# Store the datasets as Data Frame variables
trainDf = pd.read_csv('loan-train.csv')
testDF = pd.read_csv('loan-test.csv')

# View the first few rows of the Data Frame datasets
# print(trainDf.head())
# print(testDF.head())

