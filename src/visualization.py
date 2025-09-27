# Use this for your visualization functions 
import matplotlib.pyplot as plt
import plotly.express as px 
from .metrics import calculate_sma  

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

