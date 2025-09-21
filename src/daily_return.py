# Module: daily_return.py
# Author: Liao Xue E
# Date: 18/9/2025

import pandas as pd

def calculate_daily_returns(data):
    # Calculates the daily simple returns for stock data.
    # Formula: Daily Return = (Today's Close - Yesterday's Close) / Yesterday's Close    Args:
    # data (pd.DataFrame): DataFrame containing stock data with a 'Close' column
    
    try:
        data['Daily_Return'] = data['Close'].pct_change()
        
        data['Daily_Return'] = data['Daily_Return'].round(4)
        
        return data
        
    except KeyError:
        print("Error: DataFrame must contain a 'Close' column")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

