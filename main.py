# File: main.py
# import pandas as pd
from src.data_handler import data_handler
from src.visualization import plot_price_and_sma


def main():
    filepath = 'https://github.com/Eddamame/P5-4_PythonProject/blob/main/data/StockAnalysisDataset.csv?raw=true'
    # FilterName and filterTime is optional
    filterName = ['AAPL', 'AMZN', 'MSFT', 'GOOG']
    filterTime = (2016, 2017)
    df = data_handler(filepath, filterName, filterTime)
    # Get user input for the stock name and the window size
    stock_name = input("Which stock market would you like to see: ").strip()
    window_size = input("Enter SMA window size (e.g., 50): ").split(',')
    window_size = [int(x.strip()) for x in window_size]
    # Plot the price and SMA
    plot_price_and_sma(stock_name, window_size)
    print(df.head(5))
if __name__ == "__main__":
    main()