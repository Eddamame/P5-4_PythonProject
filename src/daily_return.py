# Module: daily_return.py
# Description: Calculates daily returns for stock price data.
# Author: Liao Xue E
# Date: 18/9/2025

import pandas as pd

def calculate_daily_returns(data):
    # Calculates the daily simple returns for stock data.
    # Formula: Daily Return = (Today's Close - Yesterday's Close) / Yesterday's Close    Args:
    # data (pd.DataFrame): DataFrame containing stock data with a 'Close' column
    
    try:
        data['Daily_Return'] = data['Close'].pct_change()
        
        # Round the returns to 4 decimal places for better readability
        data['Daily_Return'] = data['Daily_Return'].round(4)
        
        return data
        
    except KeyError:
        print("Error: DataFrame must contain a 'Close' column")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage (for testing):
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'Close': [100, 102, 101, 105, 107]
    })
    
    print("Original Data:")
    print(sample_data)
    
    # Calculate daily returns
    result = calculate_daily_returns(sample_data)
    
    print("\nData with Daily Returns:")
    print(result)
    
    # Manual verification for the first few values:
    # Day 1: No return (NaN) 
    # Day 2: (102-100)/100 = 0.02
    # Day 3: (101-102)/102 ≈ -0.0098
    # Day 4: (105-101)/101 ≈ 0.0396