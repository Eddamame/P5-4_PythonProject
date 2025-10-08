# Use this for your visualization functions 
import pandas as pd
import plotly.graph_objects as go
from typing import Union
import matplotlib.pyplot as plt
import plotly.express as px 
from .metrics import calculate_sma, calculate_max_profit  


# --- plot SMA using plotly ---

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

# --- plot runs using plotly ---

# green for upward, red for downward
# takes in runs_df and prices from calculate_runs
# takes in a user input value for min_length

def plot_runs(runs_df, prices, min_length=4):

    if prices.empty:
        print("No data to plot")
        return None
    
    # Create the figure
    fig = go.Figure()
    
    # Add the main price line (all data in gray)
    fig.add_trace(go.Scatter(
        x=prices['date'],
        y=prices['close'],
        mode='lines',
        line=dict(color='lightgray', width=2),
        name='Close Price',
        showlegend=True
    ))
    
    # Filter significant runs
    significant_runs = runs_df[runs_df['length'] >= min_length]
    
    # Color code only the significant runs
    for _, run in significant_runs.iterrows():
        start_idx = run['start_index']
        end_idx = run['end_index']
        
        # Get the segment data
        segment = prices.iloc[start_idx:end_idx+1]
        
        color = 'green' if run['direction'] == 'Up' else 'red'
        
        fig.add_trace(go.Scatter(
            x=segment['date'],
            y=segment['close'],
            mode='lines',
            line=dict(color=color, width=3),
            name=f"{run['direction']} Run",
            showlegend=False,
            hovertemplate=f"<b>{run['direction']} Run</b><br>" +
                         f"Length: {run['length']} days<br>" +
                         "Date: %{x}<br>" +
                         "Price: %{y:.2f}<br>" +
                         "<extra></extra>"
        ))
    
    fig.update_layout(
        title=f'Market Price Runs (Minimum Length: {min_length} days)',
        xaxis_title='Date',
        yaxis_title='Close Price',
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    
    return fig


    
# --- plot maxprofit using matplotlib ---

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