#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import time

def calculate_sma(stock_name, window_size):
    # Filter the DataFrame for the selected stock
    filtered_df = df[(df['Name'] == stock_name) & (df['year'] > 2015)].copy()
    filtered_df = filtered_df.set_index('date')
    closed_price = filtered_df['close']
    n = window_size
    i = 0
    # Initialize an empty list to store SMA
    sma=[]

    while i < len(closed_price):
        # Calculate the average of current window
        window_average = round(np.sum(closed_price[
        i:i+window_size]) / window_size, 2)
    
        # Store the average of current
        # window in moving average list
        sma.append(window_average)
    
        # Shift window to right by one position
        i += 1
    filtered_df['sma'] = sma
    return filtered_df