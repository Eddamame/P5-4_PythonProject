# Module: daily_return.py
# Author: Liao Xue E
# Date: 21/9/2025

import pandas as pd
from typing import Optional

def calculate_daily_returns(data: pd.DataFrame, stock_name: Optional[str] = None) -> Optional[pd.DataFrame]:

    # Formula: Daily Return = (Today's Close - Yesterday's Close) / Yesterday's Close    Args:
    # data (pd.DataFrame): DataFrame containing stock data with a 'Close' column
    try:
        required_cols = ['Name', 'date', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        if stock_name:
            if stock_name not in data['Name'].unique():
                raise ValueError(f"Stock '{stock_name}' not found in data")
            stock_data = data[data['Name'] == stock_name].copy()
        else:
            stock_data = data.copy()
        
        stock_data = stock_data.sort_values('date')
        
        stock_data['Daily_Return'] = stock_data['close'].pct_change().round(4)
        
        return stock_data
        
    except Exception as e:
        print(f"Error calculating daily returns: {e}")
        return None
    
