import pandas as pd
import os
from typing import List, Tuple, Optional
from flask import current_app # Crucial for finding files relative to the app root

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
        ticker (str): The single ticker requested.
        filterTime (tuple[int, int]): Optional input for time filtering.
        
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
    
    # 2. Add 'name' column and strip any "(BACKUP)" tag
    df['name'] = ticker.replace(' (BACKUP)', '').strip()
    df['name'] = df['name'].astype(str)
        
    # 3. Ensure correct data-types for numeric columns (Use coerce for robustness)
    for col in ['open','close','high','low','volume']:
         if col in df.columns:
             # Use coerce to turn bad strings (or weird floats) into NaN
             df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 4. Remove rows with any missing values *after* conversion
    df.dropna(subset=['date', 'close', 'volume'], inplace=True)

    # 5. Filter by time (years)
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
    period: str, # Period is not used for CSV filtering, but kept for function signature
    filterTime: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Loads, filters, and processes the historical backup data file (test_data.csv).
    This function implements the necessary logic to handle the file path and filter 
    by ticker, regardless of the "(BACKUP)" tag added in the session.
    """
    
    # Define the absolute path to the backup CSV file
    try:
        # Standard approach for files at the root level: (APP_ROOT)/data/test_data.csv
        backup_file_path = os.path.join(current_app.root_path, 'data', 'test_data.csv')
    except RuntimeError:
        # Fallback if current_app is not available (e.g., testing outside of app context)
        backup_file_path = os.path.join('data', 'test_data.csv')
        
    current_app.logger.info(f"DEBUG: Attempting to load and process backup data from: {backup_file_path}")
    
    try:
        # Load data from the contingency CSV
        df = pd.read_csv(backup_file_path)
    except FileNotFoundError:
        current_app.logger.error(f"Backup file not found at: {backup_file_path}")
        raise FileNotFoundError(f"Backup data file not found at: {backup_file_path}.")
    except Exception as e:
        current_app.logger.error(f"Error reading backup CSV file: {e}")
        raise Exception(f"Error reading backup CSV file: {e}")

    
    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # FIX 1: Strip '(BACKUP)' from the ticker name for filtering
    # This ensures we search for 'AAPL' even if the session ticker is 'AAPL (BACKUP)'
    clean_ticker = ticker.replace(' (BACKUP)', '').strip()
    
    # Filter specific Names
    if 'name' not in df.columns:
        raise ValueError("Backup CSV is missing the required 'name' column for ticker filtering.")
        
    # Filter by the single requested ticker
    df = df[df['name'] == clean_ticker].copy()
    
    if df.empty:
        raise ValueError(f"No data found for ticker '{clean_ticker}' in backup file.")

    # Ensure correct data-types, using COERCE for robustness against odd price strings
    df['name'] = df['name'].astype(str)
    # Convert date, using coerce to turn bad values into NaT
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


# This alias is necessary because routes.py calls this name for API data handling.
def api_data_handler(
    y_data: pd.DataFrame, 
    ticker: str, # Ticker is required to set the 'name' column
    filterTime: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Wrapper for clean_api_data used in routes.py to standardize the output.
    """
    return clean_api_data(y_data, ticker=ticker, filterTime=filterTime)