# Inside app/modules/data_fetcher.py

import yfinance as yf
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import requests 

# Define the acceptable periods sent by your Flask form and the default
VALID_PERIODS = ['1y', '2y', '3y']
DEFAULT_PERIOD = '1y'

@retry(
    wait=wait_random_exponential(min=1, max=60), 
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((
        requests.exceptions.HTTPError, 
        requests.exceptions.ConnectionError, 
        Exception
    ))
)
def get_hist_data(ticker: str, period: str):
    """
    Fetches historical market data using the yfinance period argument.
    """
    
    # --- 1. Validate and Select Period ---
    # Ensure the period passed from the form is one of the valid yfinance strings
    yf_period = period.lower() if period.lower() in VALID_PERIODS else DEFAULT_PERIOD
    
    try:
        print(f"DEBUG: Attempting fetch for {ticker} using yfinance period: {yf_period}")
        
        # --- 2. Data Fetching ---
        stock_df = yf.download(
            [ticker],
            period=yf_period,   
            interval="1d", 
            progress=False,
            ignore_tz=True 
        )

        if stock_df.empty:
            raise ValueError(f"No data returned by API for Ticker: {ticker} and Period: {period}.")

        # --- 3. Clean and Format Data ---
        if isinstance(stock_df.columns, pd.MultiIndex):
            # Flatten columns if MultiIndex (common when fetching one ticker)
            stock_df.columns = stock_df.columns.droplevel(1) 

        # Reset the index (Date) to a column named 'Date'
        stock_df = stock_df.reset_index().rename(columns={'index': 'Date'})

        print(f"Successfully fetched data for {ticker}.")
        return stock_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        raise ValueError(f"Failed to get ticker '{ticker}' reason: {e}")