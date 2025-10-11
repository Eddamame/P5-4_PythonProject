import pandas as pd
import os
from typing import List, Tuple, Optional
from flask import current_app # Import current_app for better path handling and logging

# --- 1. Function for Cleaning Live API Data (Takes a DataFrame) ---
def clean_api_data(
    df_raw: pd.DataFrame, 
    ticker: str, 
    filterTime: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Data Handler for cleaning and standardizing the DataFrame returned by yfinance API.
    
    Parameters:
        df_raw (pd.DataFrame): Raw DataFrame from get_hist_data (yfinance).
        ticker (str): The single ticker requested (used to add the 'name' column).
        filterTime (tuple[int, int]): Optional input, allows filtering by specific time (start, end) in years.
        
    Output:
        pd.DataFrame: Cleaned, filtered, and standardized DataFrame.
    """
    df = df_raw.copy()
    
    # 1. Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Ensure date column is named 'date' and set to datetime
    if 'date' not in df.columns and 'Date' in df_raw.columns:
        df = df.rename(columns={'Date': 'date'})
        
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    # 2. Add 'name' column (yfinance does not provide this column)
    df['name'] = ticker.replace(' (BACKUP)', '').strip() # Strip (BACKUP) if present
    df['name'] = df['name'].astype(str)
        
    # 3. Remove Missing Values (before numeric conversion)
    df.dropna(inplace=True)

    # 4. Ensure correct data-types for numeric columns (Using coerce for robustness)
    for col in ['open','close','high','low','volume']:
         if col in df.columns:
             # FIX: Use errors='coerce' to turn bad strings into NaN, then drop the NaN
             df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows that failed numeric conversion
    df.dropna(subset=['open','close','high','low','volume'], inplace=True)

    # 5. Filter by time (years) - this is redundant if yfinance was filtered, but safe.
    if filterTime:
        start, end = filterTime
        df = df[(df['date'].dt.year >= start) & (df['date'].dt.year <= end)]

    # 6. Ensure 2 decimal points for $
    df = df.round({col: 2 for col in ['open', 'high', 'low', 'close'] if col in df.columns})

    # 7. Sort and return required columns
    df.sort_values(by=['date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Ensure all required columns are present before returning
    required_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'name']
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA 

    return df[required_cols]


# --- 2. Function for Handling Backup CSV Data (Takes a file path and cleans it) ---
def handle_backup_csv(
    ticker: str, 
    period: str, # Period is not used for CSV filtering here, but we keep the signature for routes.py
    filterTime: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Handles the backup data path by reading the CSV from a fixed path and applying 
    the full cleaning and filtering logic.
    """
    
    # Define the absolute path to the backup CSV file
    # FIX: Use current_app.root_path to find the file correctly relative to the app root
    try:
        backup_file_path = os.path.join(current_app.root_path, 'data', 'test_data.csv')
    except RuntimeError:
        # Fallback if current_app is not available (e.g., testing outside of app context)
        backup_file_path = os.path.join('data', 'test_data.csv')
        
    print(f"DEBUG: Attempting to load and process backup data from: {backup_file_path}")
    
    try:
        # Load data from the contingency CSV
        df = pd.read_csv(backup_file_path)
    except FileNotFoundError:
        # Use flask flash/logger instead of print in a production app
        raise FileNotFoundError(f"Backup data file not found at: {backup_file_path}. Please ensure it exists in the 'data/' folder.")
    except Exception as e:
        raise Exception(f"Error reading backup CSV file: {e}")

    
    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # FIX 1: Strip '(BACKUP)' from the ticker name for filtering, 
    # as the CSV only contains 'AAPL' or 'MSFT'.
    clean_ticker = ticker.replace(' (BACKUP)', '').strip()
    
    # Filter specific Names
    if 'name' not in df.columns:
        raise ValueError("Backup CSV is missing the required 'name' column for ticker filtering.")
        
    # Filter by the single requested ticker
    df = df[df['name'] == clean_ticker].copy()
    
    if df.empty:
        raise ValueError(f"No data found for ticker '{clean_ticker}' in backup file.")

    # Remove Missing Values (Initial pass)
    df.dropna(inplace=True)

    # Ensure correct data-types, using COERCE for robustness against odd price strings
    df['name'] = df['name'].astype(str)
    # Date should be fine as 'date' column is YYYY-MM-DD
    df['date'] = pd.to_datetime(df['date'], errors='coerce') 
    
    # FIX 2: Use errors='coerce' to handle bad numeric strings, then drop rows that failed.
    for col in ['open','close','high','low','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows where date or crucial numeric columns failed conversion
    df.dropna(subset=['date', 'close', 'volume'], inplace=True)
    
    # Filter by time (years)
    if filterTime:
        start, end = filterTime
        df = df[(df['date'].dt.year >= start) & (df['date'].dt.year <= end)]

    # Ensure 2 decimal points for $
    df = df.round({'open': 2, 'high': 2, 'low': 2, 'close': 2})

    # Sort by name and date, then reset index & drop previous dataframe
    df.sort_values(by=['name', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Ensure all required columns are present before returning
    required_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'name']
    return df[required_cols]


# This alias is kept for the API route logic's use
api_data_handler = clean_api_data


# Re-implementing the original api_data_handler wrapper to use the new clean_api_data
def api_data_handler(
    y_data: pd.DataFrame, 
    ticker: str, # Ticker must be passed to add the 'name' column
    filterTime: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Wrapper for clean_api_data used in routes.py
    """
    # NOTE: Your routes.py passes the ticker to api_data_handler, but your 
    # function definition was missing it. The correct implementation should look like this:
    
    # Check if 'ticker' is present in the arguments; if not, you'll need to update routes.py 
    # to ensure the ticker is passed correctly to this function.
    
    # Assuming ticker is passed in routes.py, we call the standardized cleaner:
    return clean_api_data(y_data, ticker=ticker, filterTime=filterTime)