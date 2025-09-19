#Import Libraries
import pandas as pd
from datetime import datetime,timedelta
from src.data_handler import data_handler
import time
import numpy as np
df =data_handler('https://github.com/Eddamame/P5-4_PythonProject/blob/main/data/StockAnalysisDataset.csv?raw=true')
# Create a new column year
df['year'] = pd.DatetimeIndex(df['date']).year

# filter out the Name 
stock_name = pd.unique(df['Name'])
#user imput of the stock 
def calculate_sma(stock_name, window_size):
    # Filter the DataFrame for the selected stock
    filtered_df = df[(df['Name'] == stock_name) & (df['year'] > 2015)].copy()
    filtered_df = filtered_df.set_index('date')
    closed_price = filtered_df['close'].tolist()
  
    sma = [sum(closed_price[i:i+window_size]) / window_size 
           for i in range(len(closed_price) - window_size + 1)]

    filtered_df['sma'] = [None]*(window_size-1) + sma
    return filtered_df