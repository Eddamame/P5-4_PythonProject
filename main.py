# File: main.py
import pandas as pd
from src.data_handler import data_handler
from src.visualization import plot_price_and_sma
from src.prediction import forecast_regression, validate_regression

def main():
    filepath = 'https://github.com/Eddamame/P5-4_PythonProject/blob/main/data/StockAnalysisDataset.csv?raw=true'
    # FilterName and filterTime is optional
    filterName = ['AAPL', 'AMZN', 'MSFT', 'GOOG']
    filterTime = (2016, 2017)
    df = data_handler(filepath, filterName, filterTime)
    # Get user input for the stock name and the window size
    stock_name = input("Which stock market would you like to see: ").strip()
    window_size = int(input("Enter SMA window size (e.g., 50): "))
    # Plot the price and SMA
    plot_price_and_sma(stock_name, window_size)

    # Forecast 10 days ahead
    forecast_table = forecast_regression(df, target="close", days_ahead=7, show_graph=True)

    print("\nForecast Table:")
    print(forecast_table)

    #Validation of regression
    theta, y_test, y_pred = validate_regression(df, target="close", train_ratio=0.7)
    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
        "Error": y_test - y_pred
    })
    print(results)

if __name__ == "__main__":
    main()