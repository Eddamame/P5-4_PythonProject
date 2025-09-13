"""
metrics.py

This module contains various mathematical and statistical functions for
analyzing time-series data, particularly for financial metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt 

# load data to test 
file_path = '../data/StockAnalysisDataset.csv'
data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
data.info
print(data.columns)

# --- 1. Daily Returns Function ---
#Calculates the percentage returns from a series of prices.

def calculate_returns(data: pd.Series):
    

    # TODO: Implement the logic for calculating returns.
    data['daily_return'] = ((data['close'] - data['open']) / data['open']) * 100
    print(" \n--- DataFrame with new 'daily_return' column ---")
    print(data.head())
    pass

# --- 2. Simple Moving Average (SMA) ---

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA) for a given data series.
    
    Args:
        data (pd.Series): A pandas Series of prices.
        window (int): The number of periods for the SMA calculation.
        
    Returns:
        pd.Series: A Series containing the SMA values.
    """
    # TODO: Implement the SMA calculation using pandas.rolling().
    
    pass

# --- 3. Relative Strength Index (RSI) ---

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) for a given data series.
    
    Args:
        data (pd.Series): A pandas Series of prices.
        window (int): The number of periods for the RSI calculation (default is 14).
        
    Returns:
        pd.Series: A Series containing the RSI values.
    """
    # TODO: Implement the RSI calculation logic. This is a bit more complex.
    # It involves calculating gains, losses, and then the average of each.
    pass

# --- 4. Upward and Downward Run Analysis ---
# This is a bit more unique. Let's define what this means.
# A "run" is a sequence of consecutive increases or decreases.

def analyze_runs(data: pd.Series) -> tuple[list, list]:
    """
    Analyzes a time-series to identify and measure upward and downward runs.
    
    A "run" is a sequence of consecutive price changes in the same direction.
    
    Args:
        data (pd.Series): A pandas Series of prices.
        
    Returns:
        tuple[list, list]: A tuple containing two lists:
            - A list of upward run lengths.
            - A list of downward run lengths.
    """
    # TODO: Implement the run analysis logic.
    # You'll likely iterate through the series or use a vectorized pandas approach.
    pass

# --- 5. Main Execution Block (for testing) ---
# This block allows you to run the file directly to test the functions.

if __name__ == '__main__':
    # Create some sample data
    sample_data = pd.Series([
        10, 11, 12, 11.5, 10.8, 10.9, 11, 12, 13, 12.5,
        13, 14, 15, 14.5, 14, 13.5, 13, 12, 11.5, 12
    ])
    
    print("--- Sample Data ---")
    print(sample_data)
    print("\n" + "="*30 + "\n")

    # Test SMA function
    print("--- Testing SMA ---")
    sma_values = calculate_sma(sample_data, window=5)
    print("SMA(5):\n", sma_values)
    print("\n" + "="*30 + "\n")

    # Test RSI function
    print("--- Testing RSI ---")
    rsi_values = calculate_rsi(sample_data, window=5) # Use a smaller window for this sample
    print("RSI(5):\n", rsi_values)
    print("\n" + "="*30 + "\n")
    
    # Test Run Analysis
    print("--- Testing Run Analysis ---")
    upward_runs, downward_runs = analyze_runs(sample_data)
    print("Upward Run Lengths:", upward_runs)
    print("Downward Run Lengths:", downward_runs)
    print("\n" + "="*30 + "\n")