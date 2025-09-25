import pandas as pd
from src.data_handler import data_handler
from src.prediction import validate_regression

filepath = 'https://github.com/Eddamame/P5-4_PythonProject/blob/main/data/StockAnalysisDataset.csv?raw=true'
filterName = ['AAPL', 'AMZN', 'MSFT', 'GOOG']
filterTime = (2016, 2017)
df = data_handler(filepath, filterName, filterTime)
# Validate regression
theta, y_test, y_pred = validate_regression(df, target="close", train_ratio=0.7)

print("Coefficients:", theta)
print("Actual Test Values:", y_test)
print("Predicted Test Values:", y_pred)