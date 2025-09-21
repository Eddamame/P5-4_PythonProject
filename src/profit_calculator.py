# Module: profit_calculator.py
# Author: Liao Xue E
# Date: 20/9/2025

from typing import List, Union
import pandas as pd

def calculate_max_profit(prices: Union[List[float], pd.Series]) -> float:
    """
    Calculates maximum profit achievable through multiple buy/sell transactions
    using the Valley-Peak approach (Greedy Algorithm).
    
    Time Complexity: O(n) where n is number of prices
    Space Complexity: O(1)
    
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


