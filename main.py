# File: main.py

from src.data_handler import fetch_stock_data_from_api, preprocess_data, save_data_to_csv
import pandas as pd

def main():
    """
    The main function to run the stock data acquisition and processing pipeline.
    """
    # 1. Configuration
    api_key = 'demo'  # Replace with your actual API key
    tickers = ['IBM', 'MSFT', 'AAPL', 'GOOGL']
    output_file = 'data/stock_data.csv'

    # 2. Data Acquisition
    # Call the data handler to fetch the raw data from the API
    raw_data = fetch_stock_data_from_api(tickers, api_key)
    
    if not raw_data:
        print("No data was fetched. Exiting.")
        return
        
    # 3. Data Preprocessing
    # Call the data handler to clean and format the raw data
    stock_df = preprocess_data(raw_data)
    
    # Check if the DataFrame is empty before proceeding
    if stock_df.empty:
        print("Preprocessed DataFrame is empty. Exiting.")
        return

    # 4. Data Persistence
    # Call the data handler to save the final DataFrame to a CSV
    save_data_to_csv(stock_df, output_file)

    # 5. Display Final Results
    print("\n--- Final DataFrame ---")
    print(stock_df.head())
    print("\n--- DataFrame Information ---")
    stock_df.info()

if __name__ == "__main__":
    main()