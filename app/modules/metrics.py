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
----- SMA Analysis ------------------
Author: Si Yun
1. SMA Analysis
Input:
    df: pd.DataFrame: cleaned data from data_handler
    window_size: list of window size defined by user (datatype: int)    
return:
    dataframe of the calculated SMA 
step 1. Validate inputs
    Check that 'date' and 'close' columns exist in the DataFrame.
    Verify that each window size is a positive integer.
step 2. Prepare the data
    Set 'date' as the index.
    Retrieve the 'close' prices and convert them into a list for easier calculation.
step 3. Perform SMA calculation
    For each window size, compute the average closing price using the sliding window approach.
    The first n-1 entries will be None since there's not enough data to calculate SMA.
step 4. Store results
    Save the computed SMA values into a new column (e.g., sma_20, sma_50) in the DataFrame.
step 5. Return output
    Return the updated DataFrame containing all SMA columns.
step 6.Error handling
    Use the except block to catch and print input or unexpected errors without crashing the program.

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
def calculate_sma(df: pd.DataFrame, window_sizes: list[int]) -> pd.DataFrame:
    try:
        
        if 'date' not in df.columns:
            raise ValueError("'date' column not found in dataframe")
        if 'close' not in df.columns:
            raise ValueError("'close' column not found in dataframe")
        if not all(isinstance(n, int) and n > 0 for n in window_sizes):
            raise ValueError("window_sizes must be a positive integers.")

        df = df.set_index('date')
        close_prices = df['close'].tolist()
    
        for n in window_sizes:
            if len(close_prices) < n:
                df[f'sma_{n}'] = [None] * len(close_prices)
                continue

            sma = [None] * (n - 1)
            window_sum = sum(close_prices[:n])
            sma.append(round(window_sum / n, 2))

            for i in range(n, len(close_prices)):
                window_sum += close_prices[i] - close_prices[i - n]
                sma.append(round(window_sum / n, 2))

            df[f'sma_{n}'] = sma

        return df

    except (KeyError, ValueError) as e:
        print(f"Input Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

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


    




    

   
    
       
    
