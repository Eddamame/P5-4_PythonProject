import yfinance as yf 
import pandas as pd 

# gets historical market data for specified range and returns a pandas dataframe 

def get_hist_data(ticker, period):
    try:
        # try to download the data
        stock_df = yf.download(ticker, period=period, progress=False)

        # check if the DataFrame is empty 
        if stock_df.empty:
            raise ValueError(f"No data found for Ticker: {ticker} and Period: {period}. Check for invalid ticker or period string.")
        
        # handle Potential MultiIndex (PREVENTS TUPLE ERROR)
        # Flatten the columns if yfinance returns a MultiIndex.
        if isinstance(stock_df.columns, pd.MultiIndex):
            # Select the top level (index 0) which corresponds to 'Open', 'High', etc.
            stock_df.columns = stock_df.columns.get_level_values(0)

        # reset the index to turn the 'Date' index into a column
        stock_df = stock_df.reset_index()

        # select and return the required columns
        return stock_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
    except ValueError as e:
        # no data found
        print(f"Data Fetching Error: {e}")
        return pd.DataFrame() 
        
    except Exception as e:
        # any other unexpected network or yfinance errors
        print(f"An unexpected error occurred during data fetching: {e}")
        return pd.DataFrame()

'''
ticker = yf.Ticker("AAPL")
history = ticker.history(period ='6mo')

print(history)
'''