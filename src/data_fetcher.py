import yfinance as yf
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import requests 

# gets historical market data for specified range and returns a pandas dataframe 

# using tenacity we can retry if it fails.
@retry(
    # wait 1s, then ~2s, then ~4s, etc., up to a max of 60 seconds between retries.
    wait=wait_random_exponential(min=1, max=60), 
    
    # stop retrying after 5 attempts.
    stop=stop_after_attempt(5),
    
    retry=retry_if_exception_type((requests.exceptions.HTTPError, requests.exceptions.ConnectionError))
)
def get_hist_data(ticker, period):
    try:
        # try to download the data
        stock_df = yf.download(ticker, period=period, progress=False)

        # check if the DataFrame is empty 
        if stock_df.empty:

            # tenacity will NOT retry this. It will fail immediately
            raise ValueError(f"No data found for Ticker: {ticker} and Period: {period}. Check for invalid ticker or period string.")
        
        # handle MultiIndex (else data_handler will fail)
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = stock_df.columns.get_level_values(0)

        # reset the index to turn the 'Date' index into a column
        stock_df = stock_df.reset_index()

        # select and return the required columns
        print(f"Successfully fetched data for {ticker}.") # Added for clarity
        return stock_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
    except ValueError as e:
        # no data found for a valid ticker 
        print(f"Data Fetching Error: {e}")
        return pd.DataFrame() 
        
    except Exception as e:
        # any other unexpected errors after all retries have failed.
        print(f"An unexpected error occurred for {ticker} after several retries: {e}")
        return pd.DataFrame()