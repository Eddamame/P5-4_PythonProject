#Import Libraries
import pandas as pd
from datetime import datetime,timedelta
from src.data_handler import data_handler
import time
import numpy as np
df=data_handler('https://github.com/Eddamame/P5-4_PythonProject/blob/main/data/StockAnalysisDataset.csv?raw=true')
# Create a new column year
df['year'] = pd.DatetimeIndex(df['date']).year
# filter out the Name 
stock_name = pd.unique(df['Name'])
#user input of the stock 
def calculate_sma(stock_name, window_size):
    # Filter the DataFrame for the selected stock
    filtered_df = df[(df['Name'] == stock_name) & (df['year'] > 2015)].copy()

    # set index to date
    filtered_df = filtered_df.set_index('date')
    closed_price = filtered_df['close']
  
    i=0
    # Initialize an empty list to store simple moving averages
    sma =[]
    # Loop through the closed price to consider every window size
    while i < len(closed_price) - window_size + 1:
              # Store elements from i to i+window_size
              # in list to get the current window
              window = closed_price[i : i + window_size]
              # Calculate the average of current window
              window_average = round(sum(window) / window_size, 2)
              # Store the average of current
              # window in moving average list
              sma.append(window_average)
              # Shift window to right by one position
              i += 1
    filtered_df['sma'] = [None]*(window_size-1) + sma
    return filtered_df