# File: main.py
# import pandas as pd
#Run main to test both graphs
<<<<<<< Updated upstream
from app.modules.metrics import calculate_runs, get_significant_runs
from app.modules.visualization import plot_runs
#from app.modules.prediction import validate_and_plot, predict_next_day, plot_actual_prices
from app.modules.visualization import plot_daily_returns_plotly, plot_max_profit_segments

#from app.modules.metrics import calculate_runs, get_significant_runs
#from app.modules.visualization import plot_price_and_sma, plot_max_profit_segments, plot_runs
# from app.modules.prediction import validate_and_plot, predict_next_day, plot_actual_prices
>>>>>>> Stashed changes
from app.modules.data_fetcher import get_hist_data
from app.modules.data_handler import api_data_handler

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

<<<<<<< Updated upstream
data = get_hist_data('GM', '12mo')
df = api_data_handler(data)
runs_df, direction, prices = calculate_runs(df)
my_plot = plot_runs(runs_df, prices, 3)
if my_plot is not None:
    my_plot.show()
=======
data = get_hist_data('AAPL', '12mo')
df = api_data_handler(data)
data1 = get_hist_data('PLTR', '12mo')
df1 = api_data_handler(data1)
data2 = get_hist_data('GM', '12mo')
df2 = api_data_handler(data2)
# print(data)
# print(clean_data)
>>>>>>> Stashed changes


# Test visualizations for daily returns and max profit
df.columns = [col.lower() for col in df.columns]  
df['name'] = 'AAPL'
df1.columns = [col.lower() for col in df1.columns]  
df1['name'] = 'PLTR'
df2.columns = [col.lower() for col in df2.columns]  
df2['name'] = 'GM'

plot_daily_returns_plotly(df1, stock_name='PLTR')
# plot_max_profit_segments(df1, stock_name='PLTR')
# plot_daily_returns_plotly(df2, stock_name='GM')
# plot_max_profit_segments(df2, stock_name='GM')
# plot_daily_returns_plotly(df, stock_name='AAPL')
# plot_max_profit_segments(df, stock_name='AAPL')
# runs_df, direction, prices = calculate_runs(clean_data)
# result = get_significant_runs(runs_df, 5)
# print(result['significant_runs'] )

# my_plot = plot_runs(runs_df, prices, 6)
# if my_plot is not None:
#     my_plot.show()

# --- Model Validation ---
# Call validate_model and capture the returned actual and predicted values
# test_dates, actual_prices, predicted_prices = validate_model(df, target_column='close')

# # --- Visualization ---
# # Call the new plotting function with the captured data
# plot_prediction_vs_actual_line(test_dates, actual_prices, predicted_prices)

# # Display the comparison table
# display_prediction_comparison_table(test_dates, actual_prices, predicted_prices)

# # --- Future Forecasting ---
# forecast_prices(data=df, target_column='close')
