import yfinance as yf
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import requests 
from datetime import datetime
from dateutil.relativedelta import relativedelta 

@retry(
    # Wait 1s, then ~2s, then ~4s, etc., up to a max of 60 seconds between retries.
    wait=wait_random_exponential(min=1, max=60), 
    
    # Stop retrying after 5 attempts.
    stop=stop_after_attempt(5),
    
    # FIX: Include the general 'Exception' class to retry on yfinance internal errors 
    # like YFTzMissingError, which are not network errors.
    retry=retry_if_exception_type((requests.exceptions.HTTPError, requests.exceptions.ConnectionError, Exception))
)
def get_hist_data(ticker: str, period: str):
    """
    Fetches historical market data for a given ticker and period.
    
    The period string (e.g., '12mo') is converted into explicit start and end 
    dates using python-dateutil for reliable data retrieval via yfinance.
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
        # --- 2. Data Fetching ---
        # Use start/end dates for robustness and set ignore_tz=True to bypass 
        # yfinance's strict timezone check that causes YFTzMissingError.
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
            # This will NOT be retried by tenacity.
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
        # This catches internal errors (retried by tenacity) or the ValueError above 
        # (which is not retried if it's not the last attempt).
        # We re-raise the error to let the Flask route handle the final failure.
        raise ValueError(f"No historical data could be found for Ticker: {ticker}. Error: {e}")
