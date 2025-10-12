import pandas as pd
import os
import uuid
from typing import Optional, Dict, Tuple
from flask import current_app
from datetime import datetime, date
from dateutil.relativedelta import relativedelta 

# --- In-Memory Data Cache (Fix for Session Overflow) ---
_data_cache: Dict[str, pd.DataFrame] = {}

def store_clean_data(df: pd.DataFrame) -> str:
    """
    Stores a clean DataFrame in the in-memory cache and returns a unique key.
    """
    cache_key = str(uuid.uuid4())
    _data_cache[cache_key] = df
    try:
        current_app.logger.info(f"Stored DataFrame in cache with key: {cache_key}. Current cache size: {len(_data_cache)}")
    except RuntimeError:
        pass
    return cache_key

def retrieve_clean_data(cache_key: str) -> Optional[pd.DataFrame]:
    """
    Retrieves a clean DataFrame from the cache and immediately removes it.
    """
    df = _data_cache.pop(cache_key, None)
    try:
        current_app.logger.info(f"Retrieved and removed DataFrame for key: {cache_key}. Remaining cache size: {len(_data_cache)}")
    except RuntimeError:
        pass
    return df

# --- Helper Function for Date Calculation (Used by Backup Handler) ---

def _get_start_date_from_period(period: str) -> date:
    """Converts a period string ('1y', '2y', etc.) to a required start date."""
    today = datetime.now().date()
    
    # Use the same logic that yfinance implements
    if period == '1y':
        return today - relativedelta(years=1)
    elif period == '2y':
        return today - relativedelta(years=2)
    elif period == '3y':
        return today - relativedelta(years=3)
    
    # Fallback to a wider range if an unexpected period is passed
    return today - relativedelta(years=5) 


# --- 1. Function for Cleaning Live API Data (Simplified) ---

def api_data_handler(
    df_raw: pd.DataFrame, 
    ticker: str
) -> pd.DataFrame:
    """
    Data Handler for cleaning and standardizing the DataFrame returned by yfinance API.
    
    The raw data is assumed to be filtered for the correct period already.
    """
    df = df_raw.copy()
    
    # 1. Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Ensure date column is named 'date' and set to datetime 
    if 'date' not in df.columns and 'Date' in df_raw.columns:
        df = df.rename(columns={'Date': 'date'})
    
    if df.index.name and df.index.name.lower() in ('date', 'Date'):
        df = df.reset_index(names=['date'])
        
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    # 2. Add 'name' column and strip any "(BACKUP)" tag
    df['name'] = ticker.replace(' (BACKUP)', '').strip()
    df['name'] = df['name'].astype(str)
        
    # 3. Ensure correct data-types for numeric columns
    for col in ['open','close','high','low','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 4. Remove rows with any missing values *after* conversion
    df.dropna(subset=['date', 'close', 'volume'], inplace=True)

    # 5. Ensure 2 decimal points for $
    df = df.round({col: 2 for col in ['open', 'high', 'low', 'close'] if col in df.columns})

    # 6. Sort and return required columns
    df.sort_values(by=['date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    required_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'name']
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA 

    return df[required_cols]


# --- 2. Function for Handling Backup CSV Data (Now filters by period) ---

def handle_backup_csv(
    ticker: str, 
    period: str, # <-- CRITICAL: This parameter is now USED for filtering
    filterTime: Optional[Tuple[int, int]] = None # Kept for signature but ignored
) -> pd.DataFrame:
    """
    Loads, filters, and processes the historical backup data file, filtering
    the data based on the requested 'period' string.
    """
    
    # Define the absolute path to the backup CSV file
    try:
        backup_file_path = os.path.join(
            current_app.root_path, 
            os.pardir, 
            'data', 
            'backup_data.csv'
        )
    except RuntimeError:
        backup_file_path = os.path.join('data', 'backup_data.csv')
        
    try:
        current_app.logger.info(f"DEBUG: Attempting to load and process backup data from: {backup_file_path}")
    except RuntimeError:
        pass
    
    try:
        # --- CRITICAL FIX: Load the CSV and parse the 'date' column immediately ---
        df = pd.read_csv(
            backup_file_path,
            # Tell Pandas to parse 'date' as a datetime object
            parse_dates=['date'],   
            # Tell the parser to use Day/Month/Year (DD/MM/YYYY) format
            dayfirst=True           
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Backup data file not found at: {backup_file_path}. Please ensure it exists.")
    except Exception as e:
        raise Exception(f"Error reading backup CSV file: {e}")

    
    # Standardize column names and clean ticker
    df.columns = [col.lower() for col in df.columns]
    clean_ticker = ticker.replace(' (BACKUP)', '').strip()
    
    # Filter specific Names
    if 'name' not in df.columns:
        raise ValueError("Backup CSV is missing the required 'name' column for ticker filtering.")
        
    # Filter by the single requested ticker
    df = df[df['name'] == clean_ticker].copy()
    
    if df.empty:
        raise ValueError(f"No data found for ticker '{clean_ticker}' in backup file.")

    # Convert data types
    df['name'] = df['name'].astype(str)
    
    # The 'date' column is now already a datetime object due to pd.read_csv
    # We remove the old manual conversion: df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    
    for col in ['open','close','high','low','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows where date or crucial numeric columns failed conversion
    df.dropna(subset=['date', 'close', 'volume'], inplace=True)
    
    # Final check after cleaning
    if df.empty:
        raise ValueError(f"Ticker '{clean_ticker}' found in backup, but all rows were dropped due to bad date/numeric values.")

    # --- Filter by Period ---
    try:
        start_date = _get_start_date_from_period(period)
        # Filter the DataFrame to include only dates from the start_date onwards
        # This comparison is highly reliable when 'date' is a proper datetime object.
        df = df[df['date'].dt.date >= start_date].copy()
    except Exception as e:
         try:
            current_app.logger.warning(f"Failed to apply period filter '{period}' to backup data: {e}")
         except RuntimeError:
            pass 

    # Ensure 2 decimal points for $
    df = df.round({'open': 2, 'high': 2, 'low': 2, 'close': 2})

    # Sort by name and date, then reset index & drop previous dataframe
    df.sort_values(by=['name', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    required_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'name']
    return df[required_cols]