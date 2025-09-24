"""
metrics.py

Mathematical and statistical functions for
analyzing time-series data, particularly for financial metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt 
from data_handler import data_handler

# --- SMA Analysis ---


# --- Daily Returns --- 


# --- Profit Calculator --- 


 
# --- Upward and Downward Run Analysis ---

# First, filter out prices solely based on ticker name
# get the closing prices 

def get_closing_prices(data, ticker=None):
    """
    Returns closing prices for the whole DataFrame or for a specific ticker.
    """
    if ticker:
        filtered = data[data["Name"] == ticker]
        return filtered["close"]
    else:
        return data["close"]


    
def calculate_runs():
    # daily changes in closing prices
    changes = prices.diff()

    # upward, no change, downward (1, 0, -1)
    # limitation - mainly directional; ignores magnitude (Use RSI)
    direction = np.where(changes > 0, 1, np.where(changes < 0, -1, 0))

    # initialise run count
    runs = []
    current_run_length = 1
    
    # prevent indexerror
    current_direction = direction[0] if len(direction) > 0 else 0
    

    # check if current direction is the same as indexed direction, if it is continue run count
    for i in range(1, len(direction)):
        if direction[i] == current_direction and direction[i] != 0:
            # Continue current run
            current_run_length += 1
        else:
            # End current run, start new one
            if current_direction != 0:  # we're not tracking zero runs
                runs.append({
                    'start_date': prices.index[i - current_run_length],
                    'end_date': prices.index[i - 1],
                    'direction': 'Up' if current_direction == 1 else 'Down',
                    'length': current_run_length,
                    'start_index': i - current_run_length,
                    'end_index': i - 1
                })
            current_run_length = 1
            current_direction = direction[i]

    # Close the loop, record the final run
    if current_direction != 0:
        runs.append({
            'start_date': prices.index[len(prices) - current_run_length],
            'end_date': prices.index[-1],
            'direction': 'Up' if current_direction == 1 else 'Down',
            'length': current_run_length,
            'start_index': len(prices) - current_run_length,
            'end_index': len(prices) - 1
        })

    return pd.DataFrame(runs), direction
    

    
def get_significant_runs(runs_df, min_length=5):
    # To filter out runs based on length depending on trading methodology
    significant = runs_df[runs_df['length'] >= min_length]
    up_runs = significant[significant['direction'] == 'Up']
    down_runs = significant[significant['direction'] == 'Down']

    return {
        'up_runs': up_runs,
        'down_runs': down_runs
    }


pass

# --- TESTING ---
# This block allows you to run the file directly to test the functions.

if __name__ == '__main__':
    # Create some sample data
    filepath = ''
    sample_data = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
    

    # Test Run Analysis
    print("--- Testing Run Analysis for AMZN ---")
    prices = get_closing_prices(sample_data, "AMZN")
    
    runs_df, direction = calculate_runs() 
    # Can use this to extract notable_up_runs or notable_down_runs
    significant_runs = get_significant_runs(runs_df)
    print(f"Significant up runs: {significant_runs['up_runs']}")   
    print(f"Significant up runs: {significant_runs['down_runs']}") 
    