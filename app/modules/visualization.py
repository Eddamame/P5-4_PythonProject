# Use this for your visualization functions 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Union
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
from typing import Optional
from .metrics import calculate_sma, calculate_daily_returns, calculate_max_profit 


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


#  Prediction Visualization

# Plot actual vs. predicted values for the test set

def plot_actual_vs_predicted(actual_values, predicted_values, stock_name=""):
    """
    Creates an interactive scatter plot of actual vs. predicted values using Plotly.
    
    Parameters:
        actual_values (np.array): The true target values from the test set.
        predicted_values (np.array): The values predicted by the model for the test set.
        stock_name (str): The name of the stock for the chart title.
    """
    fig = go.Figure()

    # Add a scatter plot for the actual vs. predicted values.
    # Each point represents one prediction from the test set.
    fig.add_trace(go.Scatter(
        x=actual_values,
        y=predicted_values,
        mode='markers',
        name='Actual vs. Predicted',
        marker=dict(color='blue', opacity=0.7)
    ))

    # Add a 45-degree line representing a perfect prediction (where actual equals predicted).
    # A good model's points will be close to this line.
    min_val = min(np.min(actual_values), np.min(predicted_values))
    max_val = max(np.max(actual_values), np.max(predicted_values))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction Line',
        line=dict(color='red', dash='dash')
    ))

    # Update the layout with a title and axis labels for clarity.
    fig.update_layout(
        title=f'Model Performance for {stock_name}: Actual vs. Predicted Values (Test Set)',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        showlegend=True,
        width=800,
        height=600
    )

    fig.show()