import yfinance as yf
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import requests
import json
from typing import Literal

# Define valid periods explicitly for better type checking (optional)
YF_PERIOD = Literal['1y', '2y', '3y', '6mo', 'max']

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((
        requests.exceptions.HTTPError,
        requests.exceptions.ConnectionError,
        json.JSONDecodeError,
        Exception
    ))
)
def get_hist_data_minimal(ticker: str, period: YF_PERIOD):
    """
    Fetches historical market data for a given ticker using the yfinance 'period' parameter.
    Assumes 'period' is a valid yfinance string (e.g., '1y', '2y', '3y').
    """
    try:
        # --- DEBUGGING STEP ---
        print(f"DEBUG: Attempting fetch for {ticker} with period {period}")

        # --- 1. Data Fetching (Minimalist) ---
        # The period parameter handles the date range calculation internally.
        stock_df = yf.download(
            [ticker],       # Pass ticker as a list
            period=period,  # Use the valid yfinance period string
            interval="1d",
            progress=False,
            ignore_tz=True
        )

        if stock_df.empty:
            raise ValueError(f"No data returned by API for Ticker: {ticker} and Period: {period}.")

        # --- 2. Clean and Format Data ---

        # Handle MultiIndex (Robustness check for yfinance weirdness)
        if isinstance(stock_df.columns, pd.MultiIndex):
            # Flatten the columns.
            stock_df = stock_df.droplevel(1, axis=1)

        # Reset the index to turn the 'Date' index into a column
        stock_df = stock_df.reset_index()

        # Success: Select and return the required columns
        print(f"Successfully fetched data for {ticker}.")
        return stock_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    except Exception as e:
        # Re-raise as ValueError for consistency
        raise ValueError(f"Failed to get ticker '{ticker}' reason: {e}")