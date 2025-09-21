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
    window_size = int(input("Enter SMA window size (e.g., 50): "))
    # Plot the price and SMA
    plot_price_and_sma(stock_name, window_size)
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