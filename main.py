# File: main.py

from src.data_handler import data_handler
from src.model import slr, predict
import pandas as pd

def main():
    filepath = 'https://github.com/Eddamame/P5-4_PythonProject/blob/main/data/StockAnalysisDataset.csv?raw=true'
    # FilterName and filterTime is optional
    filterName = ['AAPL', 'AMZN', 'MSFT', 'GOOG']
    filterTime = (2016, 2017)
    df = data_handler(filepath, filterName, filterTime)
    print(df.head(5))

    x = df['low'].iloc[:-1].values.tolist()
    y = df['close'].iloc[1:].values.tolist()

    slope, intercept = slr(x, y)
    print(f"Slope: {slope:.4f}, Intercept {intercept: {intercept:.4f}}")

    last_low = df['low'].iloc[-1]
    prediction = predict([last_low], slope, intercept)[0]

    print(prediction)
if __name__ == "__main__":
    main()