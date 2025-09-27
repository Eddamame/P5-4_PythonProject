# File: main.py
import pandas as pd
from src.data_handler import data_handler
from src.visualization import plot_price_and_sma, plot_max_profit_segments, plot_predictions
from src.prediction import multiple_linear_regression


def main():
    filepath = 'https://github.com/Eddamame/P5-4_PythonProject/blob/main/data/StockAnalysisDataset.csv?raw=true'
    # FilterName and filterTime is optional
    filterName = ['AAPL', 'AMZN', 'MSFT', 'GOOG']
    filterTime = (2016, 2017)
    df = data_handler(filepath, filterName, filterTime)
    # Get user input for the stock name and the window size
    #stock_name = input("Which stock market would you like to see: ").strip()
    #window_size = int(input("Enter SMA window size (e.g., 50): "))
    # Plot the price and SMA
    #plot_price_and_sma(stock_name, window_size)
    #plot_max_profit_segments(df['close']) 

    days_ahead = 10
    # Perform multiple linear regression
    predictions = multiple_linear_regression(df, target_column='close', days_ahead=days_ahead)

    # Plot historical and predicted values
    plot_predictions(df, predictions, target_column='close')
if __name__ == "__main__":
    main()