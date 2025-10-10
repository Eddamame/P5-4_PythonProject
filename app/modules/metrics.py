"""
metrics.py

Mathematical and statistical functions for
analyzing time-series data, particularly for financial metrics.
"""

import pandas as pd
import numpy as np
from .data_handler import data_handler
import pandas as pd
from typing import Optional
from typing import List, Union

"""

----- Daily Returns ------------------
Author: Xue E
Objective:
    Compute daily returns for a given stock using cleaned data from api_data_handler.
Features:
    - Input: Pandas DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume', 'name'].
    - Filters data by stock name and optional date range.
    - Calculates percentage change in closing prices.
Target:
    - Output: DataFrame with columns ['date', 'close', 'Daily_Return'].
Steps:
    1. Filter data for the specified stock name.
    2. Apply optional start_date and end_date filters.
    3. Sort data by date to ensure chronological order.
    4. Compute daily returns using pandas pct_change() and round to 4 decimals.
    5. Return DataFrame with date, close, and Daily_Return columns.

----- Max Profit ----------------------
Author: Xue E
Objective:
    Calculate the maximum achievable profit using multiple buy-sell transactions (Valley–Peak strategy).
Features:
    - Input: Pandas DataFrame from api_data_handler with columns ['date', 'close', 'name'].
    - Filters data by stock name and optional date range.
    - Computes profit by summing all positive differences between consecutive closing prices.
Target:
    - Output: Float representing total maximum profit.
Steps:
    1. Filter data for the specified stock name.
    2. Apply optional start_date and end_date filters.
    3. Extract closing prices as a list.
    4. Iterate through prices and sum all positive differences.
    5. Return the total profit rounded to 2 decimals.

"""

# --- SMA Analysis ---
df =data_handler('https://github.com/Eddamame/P5-4_PythonProject/blob/main/data/StockAnalysisDataset.csv?raw=true')
# Create a new column year
df['year'] = pd.DatetimeIndex(df['date']).year
# filter out the Name 
stock_name = pd.unique(df['name'])
def calculate_sma(stock_name, window_sizes):
    filtered_df = df[(df['name'] == stock_name) & (df['year'] > 2015)].copy()
    filtered_df = filtered_df.set_index('date')
    closed_price = filtered_df['close']
    for n in window_sizes:
        sma = []
        for i in range(len(closed_price)):
                if i < n - 1:
                    sma.append(None)  # Always create the column
                else:
                    window = closed_price[i - n + 1 : i + 1]
                    sma.append(round(sum(window)/n, 2))
        filtered_df[f'sma_{n}'] = sma  # Column always exists

    return filtered_df

# --- Daily Returns --- 
def calculate_daily_returns(data):

    try:
        # Check if required columns exist
        if 'date' not in data.columns:
            raise ValueError("'date' column not found in dataframe")
        if 'close' not in data.columns:
            raise ValueError("'close' column not found in dataframe")
        
        # Select required columns and copy
        stock_data = data[['date', 'close']].copy()

         # Convert date to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(stock_data['date']):
            stock_data['date'] = pd.to_datetime(stock_data['date'])
        
        # Check if we have data
        if len(stock_data) == 0:
            raise ValueError("No data available")
        
        # Ensure data is sorted by date
        stock_data = stock_data.sort_values('date')
        
        # Calculate daily returns using percentage change
        stock_data['Daily_Return'] = stock_data['close'].pct_change().round(4)
        
        return stock_data[['date', 'close', 'Daily_Return']]
    
    except Exception as e:
        print(f"Error calculating daily returns: {e}")
        return pd.DataFrame()


# --- Profit Calculator --- 
def calculate_max_profit(data):
    """
    Calculates maximum profit achievable through multiple buy/sell transactions
    using the Valley-Peak approach (Greedy Algorithm).  
    """
    try:
        # Check if required columns exist
        if 'close' not in data.columns:
            raise ValueError("'close' column not found in dataframe")
        
        prices = data['close'].tolist()

        if len(prices) < 2:
            return 0.0
        
        # Valley–Peak algorithm: sum all positive differences
        max_profit = sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))
        
        return round(max_profit, 2)
    
    except Exception as e:
        print(f"Error calculating max profit: {e}")
        return 0.0
 
# --- Upward and Downward Run Analysis ---
# older version 
# def old_calculate_runs(data):
#     try:
#         # Check if required columns exist
#         if 'date' not in data.columns:
#             raise ValueError("'date' column not found in dataframe")
#         if 'close' not in data.columns:
#             raise ValueError("'close' column not found in dataframe")
        
#         # Select required columns and copy
#         prices = data[['date', 'close']].copy()
        
#         # Convert date to datetime if not already
#         if not pd.api.types.is_datetime64_any_dtype(prices['date']):
#             prices['date'] = pd.to_datetime(prices['date'])
        
#         # Check if we have data
#         if len(prices) == 0:
#             raise ValueError("No data available")
        
#         # Calculate daily changes
#         close_changes = prices['close'].diff()
        
#         # Convert to direction: 1 (up), -1 (down), 0 (no change)
#         direction = np.sign(close_changes)
#         direction = direction.fillna(0).astype(int)
        
#         # Initialize run tracking
#         runs = []
#         current_run_length = 1
#         current_direction = direction.iloc[0]
        
#         # Iterate through directions to find runs
#         for i in range(1, len(direction)):
#             if direction.iloc[i] == current_direction and current_direction != 0:
#                 current_run_length += 1
#             else:
#                 # Record the completed run
#                 if current_direction != 0:
#                     start_idx = i - current_run_length
#                     end_idx = i - 1
                    
#                     runs.append({
#                         'start_date': prices.iloc[start_idx]['date'],
#                         'end_date': prices.iloc[end_idx]['date'],
#                         'direction': 'Up' if current_direction == 1 else 'Down',
#                         'length': current_run_length,
#                         'start_index': start_idx,
#                         'end_index': end_idx
#                     })
                
#                 current_run_length = 1
#                 current_direction = direction.iloc[i]
        
#         # Record the final run
#         if current_direction != 0:
#             start_idx = len(prices) - current_run_length
#             end_idx = len(prices) - 1
            
#             runs.append({
#                 'start_date': prices.iloc[start_idx]['date'],
#                 'end_date': prices.iloc[end_idx]['date'],
#                 'direction': 'Up' if current_direction == 1 else 'Down',
#                 'length': current_run_length,
#                 'start_index': start_idx,
#                 'end_index': end_idx
#             })
        
#         return pd.DataFrame(runs), direction, prices
        
#     except Exception as e:
#         print(f"Error in calculate_runs: {e}")
#         return pd.DataFrame(), np.array([]), pd.DataFrame()

def calculate_runs(data):

    try:
        #pre-validation 
        if 'date' not in data.columns or 'close' not in data.columns:
            raise ValueError("'date' and 'close' columns are required.")
            
        df = data[['date', 'close']].copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # makes the original row numbers accessible for aggregation
        df = df.reset_index()
        
        if df.empty:
            # Return empty structures that match the success case
            return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

        # calculate direction 
        direction = df['close'].diff().pipe(np.sign).fillna(0).astype(int)

        # identify runs 
        run_id = direction.ne(direction.shift()).cumsum()

        # aggregate using groupby
        is_run = direction != 0
        
        runs = df[is_run].groupby(run_id[is_run]).agg(
            start_date=('date', 'first'),
            end_date=('date', 'last'),
            length=('date', 'size'),
            start_index=('index', 'first'),
            end_index=('index', 'last')
        )
        
        runs['direction'] = direction[is_run].groupby(run_id[is_run]).first().map({1: 'Up', -1: 'Down'})
        
        return runs.reset_index(drop=True), direction, df

    except Exception as e:
        print(f"Error in calculate_runs_optimized: {e}")
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()
    
# this is a quick view of runs that have reached min length 
def get_significant_runs(runs_df, min_length=4):
    #Filter runs by minimum length
    significant = runs_df[runs_df['length'] >= min_length]
    up_runs = significant[significant['direction'] == 'Up']
    down_runs = significant[significant['direction'] == 'Down']

    return {
        'up_runs': up_runs,
        'down_runs': down_runs,
        'significant_runs': significant 
    }


# --- TESTING ---
# This block allows you to run the file directly to test the functions.

# if __name__ == '__main__':
#    data = get_hist_data('AMZN', '12mo')
#    df = api_data_handler(data)

#     # Test Run Analysis
#     print("--- Testing Run Analysis for AMZN ---")

#    runs_df, direction, prices = calculate_runs(df)
## Can use this to quick view the significant runs
#    result = get_significant_runs(runs_df, 5)
#    print(result['significant_runs'] )


    




    

   
    
       
    
