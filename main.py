# File: main.py

from src.data_handler import data_handler
import pandas as pd

def main():
    filepath = 'https://github.com/Eddamame/P5-4_PythonProject/blob/main/data/StockAnalysisDataset.csv?raw=true'
    # FilterName and filterTime is optional
    filterName = ['AAPL', 'AMZN', 'MSFT', 'GOOG']
    filterTime = (2016, 2017)
    df = data_handler(filepath, filterName, filterTime)
    print(df.head(5))
if __name__ == "__main__":
    main()