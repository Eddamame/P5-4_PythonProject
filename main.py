# File: main.py
# import pandas as pd
# from src.data_handler import data_handler
# from src.visualization import plot_price_and_sma, plot_max_profit_segments
# from src.prediction import validate_and_plot, predict_next_day, plot_actual_prices
from app.modules.data_fetcher import get_hist_data
# def main():
#     filepath = 'https://github.com/Eddamame/P5-4_PythonProject/blob/main/data/StockAnalysisDataset.csv?raw=true'
#     # FilterName and filterTime is optional
#     filterName = ['AAPL']
#     filterTime = (2016, 2017)
#     df = data_handler(filepath, filterName, filterTime)
#     # Get user input for the stock name and the window size
#     stock_name = input("Which stock market would you like to see: ").strip()
#     window_size = input("Enter SMA window size (e.g., 50): ").split(',')
#     window_size = [int(x.strip()) for x in window_size]
#     # Plot the price and SMA
#     plot_price_and_sma(stock_name, window_size)
#     plot_max_profit_segments(df['close'])

#     # Plot actual stock prices
#     plot_actual_prices(df, target_column='close')
    
#     # Validate the model and plot actual vs. predicted values
#     validate_and_plot(df, target_column='close')

#     # Predict the next day's value
#     predict_next_day(df, target_column='close')

# if __name__ == "__main__":
#     main()

data = get_hist_data('AAPL', '3mo')
print(data)