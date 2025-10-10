import yfinance as yf
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import requests 
from datetime import datetime
from dateutil.relativedelta import relativedelta 
import json

@retry(
    # Wait 1s, then ~2s, then ~4s, etc., up to a max of 60 seconds between retries.
    wait=wait_random_exponential(min=1, max=60), 
    
    # Stop retrying after 5 attempts.
    stop=stop_after_attempt(5),
    
    # Retry on network errors, JSON parsing errors, and general exceptions (for YFTzMissingError).
    retry=retry_if_exception_type((
        requests.exceptions.HTTPError, 
        requests.exceptions.ConnectionError, 
        json.JSONDecodeError,
        Exception
    ))
)
def get_hist_data(ticker: str, period: str):
    """
    Fetches historical market data for a given ticker and period.
    
    Converts the period string (e.g., '12mo') into explicit start and end 
    dates for reliable data retrieval via yfinance.
    """
    
    # --- 1. Calculate explicit Start and End Dates ---
    today = datetime.now().date()
    
    # Calculate start date based on the period string
    if period == '12mo':
        start_date = today - relativedelta(years=1)
    elif period == '24mo':
        start_date = today - relativedelta(years=2)
    elif period == '36mo':
        start_date = today - relativedelta(years=3)
    else:
        # Default to 1 year if an unexpected period string is passed
        start_date = today - relativedelta(years=1)
        
    end_date = today
    
    try:
        # --- DEBUGGING STEP ---
        print(f"DEBUG: Attempting fetch for {ticker} from {start_date} to {end_date}")
        
        # --- 2. Data Fetching ---
        # Use start/end dates for robustness and set ignore_tz=True
        stock_df = yf.download(
            ticker, 
            start=start_date, 
            end=end_date, 
            interval="1d", 
            progress=False,
            ignore_tz=True 
        )

        if stock_df.empty:
            # If yfinance returns an empty DataFrame, raise a ValueError.
            # This triggers the RetryError if it happens 5 times.
            raise ValueError(f"No data returned by API for Ticker: {ticker} and Period: {period}.")

        # --- 3. Clean and Format Data ---
        # Handle MultiIndex
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = stock_df.columns.get_level_values(0)

        # reset the index to turn the 'Date' index into a column
        stock_df = stock_df.reset_index()

        # Success
        print(f"Successfully fetched data for {ticker}.")
        return stock_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        # This catches all retryable exceptions, but if the final attempt fails,
        # it re-raises as a ValueError for logging in routes.py
        raise ValueError(f"Failed to get ticker '{ticker}' reason: {e}")
