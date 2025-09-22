# Author: Lideon
# Date: 21-09-2025
# Description: This module contains functions for validating all of the members' functions and user inputs etc.

import pandas as pd
import numpy as np
import sys  
import os
from datetime import datetime, timedelta

# Importing all other team members' modules
try:
    from src.data_handler import data_handler
    from src.sma import calculate_sma
    from src.metrics import get_closing_prices, get_significant_runs   
    from src.daily_return import calculate_daily_returns
    from src.profit_calculator import calculate_max_profit
    print("All team members' modules are imported successfully and validated!")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all team members' modules are in the 'src' directory and named correctly.")

# Test data setup (global)
def create_test_data():
    # Creating a controlled dataset for testing
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')

    test_data = []

    # AAPL data with some flunctuations
    aapl_prices = [150, 152, 151, 153, 155, 154, 156, 158, 157, 159, 160, 162, 161, 163, 165, 164, 166]

    for i, date in enumerate(dates):
        price = aapl_prices[i]
        test_data.append({
            'Name': 'AAPL', 
            'date': date, 
            'open': price * 0.995, 
            'high': price * 1.02, 
            'low': price * 0.98,
            'close': price,
            'volume': 1000000 + i * 50000
            })



