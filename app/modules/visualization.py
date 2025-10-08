# Use this for your visualization functions 
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
from typing import Optional
from .metrics import calculate_sma, calculate_daily_returns, calculate_max_profit 


def plot_price_and_sma(stock_name, window_size):
    df = calculate_sma(stock_name, window_size)
    # Pick only needed columns
    cols = ['close'] + [f'sma_{w}' for w in window_size]
    df_plot = df[cols].reset_index()

    # Reshape into long format
    df_melt = df_plot.melt(id_vars="date", var_name="Series", value_name="Price")

    # Plot line chart
    fig = px.line(df_melt, x="date", y="Price", color="Series",
                  title=f"Stock Price with SMAs for {stock_name}")

    fig.show()


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

def plot_daily_returns_plotly(data: pd.DataFrame, stock_name: str,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None):
    """
    Create an interactive bar chart of daily returns using Plotly.

    Args:
        data (pd.DataFrame): Full dataset.
        stock_name (str): Stock to visualize.
        start_date (str, optional): Start date filter.
        end_date (str, optional): End date filter.
    """
    # Get filtered data with daily returns
    filtered = calculate_daily_returns(data, stock_name, start_date, end_date)

    # Create bar chart with color coding (green for positive, red for negative)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filtered['date'],
        y=filtered['Daily_Return'] * 100,  # Convert to percentage
        marker_color=['green' if val >= 0 else 'red' for val in filtered['Daily_Return']],
        name='Daily Return (%)'
    ))

    # Add chart title and labels
    fig.update_layout(title=f"Daily Returns for {stock_name}",
                      xaxis_title="Date", yaxis_title="Return (%)",
                      template="plotly_white")
    fig.show()

def plot_max_profit_segments(data, stock_name, start_date=None, end_date=None):
    """
    Plot the stock price series and highlight all buy–sell segments
    that contribute to the maximum profit (Valley–Peak strategy),
    while calling calculate_max_profit to display the total.
    """
    
    stock_data = data[data['name'] == stock_name].copy()

    if start_date:
        stock_data = stock_data[stock_data['date'] >= pd.to_datetime(start_date)]
    if end_date:
        stock_data = stock_data[stock_data['date'] <= pd.to_datetime(end_date)]

    prices = stock_data['close'].reset_index(drop=True)
    total_profit = calculate_max_profit(data, stock_name, start_date, end_date)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['date'], y=prices, mode='lines', name='Price'))
    fig.update_layout(title=f"Max Profit Segments — Total Profit: {total_profit}",
                      xaxis_title="Date", yaxis_title="Price ($)",
                      template="plotly_white")
    fig.show()
