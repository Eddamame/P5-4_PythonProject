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
def calculate_daily_returns(data: pd.DataFrame, stock_name: str,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> pd.DataFrame:

    # Formula: Daily Return = (Today's Close - Yesterday's Close) / Yesterday's Close    Args:
    # data (pd.DataFrame): DataFrame containing stock data with a 'Close' column
    try:
        stock_data = data[data['name'] == stock_name].copy()

        if start_date:
            stock_data = stock_data[stock_data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            stock_data = stock_data[stock_data['date'] <= pd.to_datetime(end_date)]
        
        stock_data = stock_data.sort_values('date')
        
        # Calculate daily returns using percentage change
        stock_data['Daily_Return'] = stock_data['close'].pct_change().round(4)
        
        return stock_data[['date', 'close', 'Daily_Return']]
    
    except Exception as e:
        print(f"Error calculating daily returns: {e}")
        return pd.DataFrame()


# --- Profit Calculator --- 
def calculate_max_profit(data: pd.DataFrame, stock_name: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> float:
    """
    Calculates maximum profit achievable through multiple buy/sell transactions
    using the Valley-Peak approach (Greedy Algorithm).
    
    Args:
        prices: List or Series of stock prices
    
    Returns:
        Maximum achievable profit
    """
    try:
        stock_data = data[data['name'] == stock_name].copy()

        if start_date:
            stock_data = stock_data[stock_data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            stock_data = stock_data[stock_data['date'] <= pd.to_datetime(end_date)]
        
        prices = stock_data['close'].tolist()

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


    




    

   
    
       
    
