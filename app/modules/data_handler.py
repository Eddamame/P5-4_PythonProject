import pandas as pd
import os
import uuid
from typing import Optional, Dict, Tuple
from flask import current_app
from datetime import datetime, date


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


# --- 1. Function for Cleaning Live API Data ---

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
    
    # 4. Remove rows with any missing values after conversion
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


# --- 2. Function for Handling Backup CSV Data ( filters by period) ---

def handle_backup_csv(
    ticker: str, 
    period: str, 
    filterTime: Optional[Tuple[int, int]] = None 
) -> pd.DataFrame:
    """
    Loads, filters, and processes the historical backup data file, filtering
    the data based on the requested 'period' string (e.g., '1y', '2y', '3y').
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
    
    # The 'unsupported operand type(s) for -: 'str' and 'DateOffset'' error
    # means latest_date (df['date'].max()) is a string/object, not a datetime.
    df['date'] = pd.to_datetime(df['date'], errors='coerce') 
    # --- END FIX ---

    for col in ['open','close','high','low','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows where date or crucial numeric columns failed conversion
    df.dropna(subset=['date', 'close', 'volume'], inplace=True)
    
    # Final check after cleaning
    if df.empty:
        raise ValueError(f"Ticker '{clean_ticker}' found in backup, but all rows were dropped due to bad date/numeric values.")

    # We use a nested try/except to specifically target issues in date filtering
    try:
        # 1. Determine the period in years (default to 5 years if period is invalid/missing)
        period_years = 0
        if period.lower().endswith('y'):
            try:
                period_years = int(period[:-1])
            except ValueError:
                # Fallback if period is '1z' or 'ay'
                period_years = 5 
        
        if period_years == 0:
            # Fallback if period is '1' or '2' (missing the 'y')
            period_years = 5 
            
        
        # 2. Calculate the start date using the latest date in the loaded dataframe (best practice)
        latest_date = df['date'].max()
        
        # Guard against NaT (Not a Time) in case of unexpected data issues
        if pd.isna(latest_date):
            raise ValueError("Maximum date value in the dataset is invalid (NaT). Cannot calculate cutoff date.")

        # Use pd.DateOffset to reliably calculate the cutoff time
        start_date_cutoff = latest_date - pd.DateOffset(years=period_years)
        
        # 3. Filter the DataFrame
        df = df[df['date'] >= start_date_cutoff].copy()
        
    except Exception as e:
        # Log the specific failure reason
        try:
            current_app.logger.warning(
                f"Date filtering failed for period '{period}' (interpreted as {period_years} years). "
                f"Using full available dataset. Error: {e}"
            )
        except RuntimeError:
            pass # Ignore if not running within Flask app context

    # Ensure 2 decimal points for $
    df = df.round({'open': 2, 'high': 2, 'low': 2, 'close': 2})

    # Sort by name and date, then reset index & drop previous dataframe
    df.sort_values(by=['name', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    required_cols = ['date', 'open', 'close', 'high', 'low', 'volume', 'name']
    return df[required_cols]
