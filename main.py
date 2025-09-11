# File: main.py

from src.data_handler import data_handler
import pandas as pd

def main():
    filepath = 'https://github.com/Eddamame/P5-4_PythonProject/blob/main/data/StockAnalysisDataset.csv?raw=true'
    filterName = ['AAPL', 'AMZN', 'MSFT', 'GOOG']
    df = data_handler(filepath, filterName)
    print(df.head(5))
if __name__ == "__main__":
    main()