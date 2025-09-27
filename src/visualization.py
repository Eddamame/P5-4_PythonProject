# Use this for your visualization functions 
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
from src.sma import calculate_sma  
from src.metrics import calculate_max_profit

def plot_price_and_sma(stock_name, window_size):
    # Calculate SMA 
    filtered_df = calculate_sma(stock_name, window_size)
    # give a dimensions to a width of 10 inches and a height of 6 inches
    plt.figure(figsize=(10, 6))
      
    # Plot close price
    plt.plot(filtered_df.index, filtered_df['close'],label='Close Price', color='blue')
    
    # Plot SMA
    plt.plot(filtered_df.index, filtered_df['sma'], label=f'SMA {window_size}', color='green')
    
    # Label the graph and provide title
    plt.title(f"Stock Price with SMA{window_size} for {stock_name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


'''
takes in 3 arguments 
- prices comes from get_closing_prices(),
- runs_df comes from calculate_runs(), 
- min_length is up to user 
'''

def plot_runs(prices, runs_df, min_length=5):
    
    plt.figure(figsize=(14, 8))
    
    # Plot price line
    plt.plot(prices.index, prices.values, 'black', linewidth=1, alpha=0.7)
    
    # Filter significant runs
    significant_runs = runs_df[runs_df['length'] >= min_length]
    
    # Draw colored lines for significant runs
    for i, run in significant_runs.iterrows():
        start_idx = run['start_index']
        end_idx = run['end_index']
        
        # Get the price segment for this run
        run_dates = prices.index[start_idx:end_idx+1]
        run_prices = prices.iloc[start_idx:end_idx+1]
        
        # Choose color and draw thick line
        color = 'green' if run['direction'] == 'Up' else 'red'
        plt.plot(run_dates, run_prices, color=color, linewidth=4, alpha=0.8)
        
        # Add run length label
        mid_idx = start_idx + (end_idx - start_idx) // 2
        plt.annotate(f"{run['length']}", 
                    xy=(prices.index[mid_idx], prices.iloc[mid_idx]),
                    xytext=(0, 15), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    plt.title(f'Stock Runs Analysis (≥{min_length} days)', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    
    # Simple legend
    if len(significant_runs[significant_runs['direction'] == 'Up']) > 0:
        plt.plot([], [], 'green', linewidth=4, label='Up Runs')
    if len(significant_runs[significant_runs['direction'] == 'Down']) > 0:
        plt.plot([], [], 'red', linewidth=4, label='Down Runs')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_max_profit_segments(prices: Union[pd.Series, list]):
    """
    Plot the stock price series and highlight all buy–sell segments
    that contribute to the maximum profit (Valley–Peak strategy),
    while calling calculate_max_profit to display the total.
    """
    # Ensure a pandas Series for easy indexing
    if isinstance(prices, list):
        prices = pd.Series(prices, index=range(len(prices)))

    if len(prices) < 2:
        raise ValueError("Need at least 2 price points to compute profit")

    # ----- total profit from your existing function -----
    total_profit = calculate_max_profit(prices)

    # ----- identify buy/sell segments (valley–peak) -----
    segments = []
    i = 0
    while i < len(prices) - 1:
        while i < len(prices) - 1 and prices.iloc[i + 1] <= prices.iloc[i]:
            i += 1
        valley = i
        while i < len(prices) - 1 and prices.iloc[i + 1] >= prices.iloc[i]:
            i += 1
        peak = i
        if peak > valley:
            segments.append((valley, peak))

    # ----- plotting -----
    plt.figure(figsize=(12, 6))
    plt.plot(prices.index, prices.values, color='black', linewidth=1, alpha=0.7, label="Price")

    for start, end in segments:
        plt.plot(prices.index[start:end + 1],
                 prices.iloc[start:end + 1],
                 color='green', linewidth=3, alpha=0.8)

        plt.scatter(prices.index[start], prices.iloc[start], color='blue', marker='^', s=80, label='Buy' if start == segments[0][0] else "")
        plt.scatter(prices.index[end],   prices.iloc[end],   color='red',  marker='v', s=80, label='Sell' if start == segments[0][0] else "")

        mid = start + (end - start)//2
        profit_segment = prices.iloc[end] - prices.iloc[start]
        plt.annotate(f"+{profit_segment:.2f}",
                     xy=(prices.index[mid], prices.iloc[mid]),
                     xytext=(0, 15), textcoords='offset points',
                     ha='center', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.6))

    plt.title(f"Max Profit Segments — Total Profit: {total_profit:.2f}")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_predictions(data, predictions, target_column):
    """
    Plot the historical data and future predictions.

    Args:
    - data (pd.DataFrame): Historical stock data with a 'date' column.
    - predictions (pd.DataFrame): Future predictions with 'date' and target column.
    - target_column (str): The column name to plot (e.g., 'close').
    """
    # Combine the last 30 days of historical data with predictions
    plot_data = data[-30:]  # Last 30 days
    plt.figure(figsize=(12, 6))

    # Plot historical data
    plt.plot(plot_data['date'], plot_data[target_column], label="Historical Data", marker='o')

    # Plot predictions
    plt.plot(predictions['date'], predictions[target_column], label="Predictions", marker='x', linestyle='--')

    # Add labels and legend
    plt.title("Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel(target_column.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()