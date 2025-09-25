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

# def get_closing_prices(data, ticker=None):
#     """
#     Returns closing prices for the whole DataFrame or for a specific ticker.
#     """
#     if ticker:
#         filtered = data[data["Name"] == ticker]
#         return filtered["close"]
#     else:
#         return data["close"]


def calculate_runs(data):
    try:
        # Check if required columns exist
        if 'date' not in data.columns:
            raise ValueError("'date' column not found in dataframe")
        if 'close' not in data.columns:
            raise ValueError("'close' column not found in dataframe")
        
        # Select the required columns
        prices = data[['date', 'close']].copy()
        
        # Ensure we have data
        if len(prices) == 0:
            raise ValueError("No data available")
        
        # Calculate daily changes in closing prices (only on the 'close' column)
        close_changes = prices['close'].diff()
        
        # Convert to direction: upward, no change, downward (1, 0, -1)
        direction = np.where(close_changes > 0, 1, np.where(close_changes < 0, -1, 0))
        
        # Initialize run tracking
        runs = []
        current_run_length = 1
        
        # Prevent IndexError
        current_direction = direction[0] if len(direction) > 0 else 0
        
        # Iterate through directions to find runs
        for i in range(1, len(direction)):
            if direction[i] == current_direction and direction[i] != 0:
                # Continue current run
                current_run_length += 1
            else:
                # End current run, start new one
                if current_direction != 0:  # Don't track zero runs
                    start_idx = i - current_run_length
                    end_idx = i - 1
                    
                    runs.append({
                        'start_date': prices.iloc[start_idx]['date'],
                        'end_date': prices.iloc[end_idx]['date'],
                        'direction': 'Up' if current_direction == 1 else 'Down',
                        'length': current_run_length,
                        'start_index': start_idx,
                        'end_index': end_idx
                    })
                
                current_run_length = 1
                current_direction = direction[i]
        
        # Close the loop, record the final run
        if current_direction != 0:
            start_idx = len(prices) - current_run_length
            end_idx = len(prices) - 1
            
            runs.append({
                'start_date': prices.iloc[start_idx]['date'],
                'end_date': prices.iloc[end_idx]['date'],
                'direction': 'Up' if current_direction == 1 else 'Down',
                'length': current_run_length,
                'start_index': start_idx,
                'end_index': end_idx
            })
        
        return pd.DataFrame(runs), direction
        
    except KeyError as e:
        print(f"Column error: {e}")
        return pd.DataFrame(), []
    except ValueError as e:
        print(f"Data error: {e}")
        return pd.DataFrame(), []
    except Exception as e:
        print(f"Unexpected error in calculate_runs: {e}")
        return pd.DataFrame(), []
  

    

    
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
    file_path = 'https://github.com/Eddamame/P5-4_PythonProject/blob/main/data/StockAnalysisDataset.csv?raw=true'
    filterName = ['AMZN']
    data = data_handler(file_path, filterName)
    

    # Test Run Analysis
    print("--- Testing Run Analysis for AMZN ---")
    
    runs_df, direction = calculate_runs(data) 
    #all runs
    print(runs_df)
    #Can use this to extract significant up or down runs
    significant_runs = get_significant_runs(runs_df, 7)
    print(f"Significant up runs: {significant_runs['up_runs']}")   
    print(f"Significant up runs: {significant_runs['down_runs']}") 
    