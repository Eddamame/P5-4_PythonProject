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
def calculate_daily_returns(data: pd.DataFrame, stock_name: Optional[str] = None) -> Optional[pd.DataFrame]:

    # Formula: Daily Return = (Today's Close - Yesterday's Close) / Yesterday's Close    Args:
    # data (pd.DataFrame): DataFrame containing stock data with a 'Close' column
    try:
        required_cols = ['name', 'date', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        if stock_name:
            if stock_name not in data['name'].unique():
                raise ValueError(f"Stock '{stock_name}' not found in data")
            stock_data = data[data['name'] == stock_name].copy()
        else:
            stock_data = data.copy()
        
        stock_data = stock_data.sort_values('date')
        
        stock_data['Daily_Return'] = stock_data['close'].pct_change().round(4)
        
        return stock_data
        
    except Exception as e:
        print(f"Error calculating daily returns: {e}")
        return None

# --- Profit Calculator --- 
def calculate_max_profit(prices: Union[List[float], pd.Series]) -> float:
    """
    Calculates maximum profit achievable through multiple buy/sell transactions
    using the Valley-Peak approach (Greedy Algorithm).
    
    Args:
        prices: List or Series of stock prices
    
    Returns:
        Maximum achievable profit
    """
    # Input validation and conversion
    if not prices:
        raise ValueError("Price list cannot be empty")
    
    if isinstance(prices, pd.Series):
        prices = prices.tolist()
    
    if len(prices) < 2:
        return 0.0  

    if any(price < 0 for price in prices):
        raise ValueError("Prices cannot be negative")
    
    max_profit = 0.0
    
    # Implement Valley-Peak algorithm
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            max_profit += prices[i] - prices[i-1]
    
    return round(max_profit, 2)
 
# --- Upward and Downward Run Analysis ---

def calculate_runs(data):
    try:
        # Check if required columns exist
        if 'date' not in data.columns:
            raise ValueError("'date' column not found in dataframe")
        if 'close' not in data.columns:
            raise ValueError("'close' column not found in dataframe")
        
        # Select the required columns and copy
        prices = data[['date', 'close']].copy()
        
        # Check if we have data
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


# --- TESTING ---
# This block allows you to run the file directly to test the functions.

# if __name__ == '__main__':
#     # Create some sample data
#     filepath = ''
#     sample_data = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
    

#     # Test Run Analysis
#     print("--- Testing Run Analysis for AMZN ---")
#     prices = get_closing_prices(sample_data, "AMZN")
    
#     runs_df, direction = calculate_runs() 
#     # Can use this to extract notable_up_runs or notable_down_runs
#     significant_runs = get_significant_runs(runs_df)
#     print(f"Significant up runs: {significant_runs['up_runs']}")   
#     print(f"Significant up runs: {significant_runs['down_runs']}") 
    
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
    
       
    
