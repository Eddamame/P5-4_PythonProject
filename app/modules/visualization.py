import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Union
import plotly.express as px 
from typing import Optional
from .metrics import calculate_sma, calculate_daily_returns, calculate_max_profit 

# --- plot SMA using plotly ---
def plot_price_and_sma(df, window_size):
    """
    Creates a Plotly figure showing Close Price and Simple Moving Averages (SMAs).
    
    Parameters:
        df (pd.DataFrame): DataFrame containing stock data, potentially with SMA columns.
        window_size (int or list[int]): Single window size or list of window sizes for SMA calculation.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure object.
    """
    try:
        # --- Handle both int and list inputs ---
        if isinstance(window_size, int):
            window_sizes = [window_size]
        elif isinstance(window_size, list):
            if not all(isinstance(w, int) and w > 0 for w in window_size):
                raise ValueError("All window sizes must be positive integers.")
            window_sizes = window_size
        else:
            raise TypeError("window_size must be an int or a list of ints.")

        # --- Ensure SMA columns exist ---
        for w in window_sizes:
            sma_col = f'sma_{w}'
            if sma_col not in df.columns:
                df = calculate_sma(df, w)
                # Assuming calculate_sma handles calculation and returns updated df
                # If it doesn't, this needs correction based on actual implementation.
                # For now, rely on external calculation/column existence.
                pass 

        # --- Create Plotly figure ---
        fig = go.Figure()

        # Add closing price line
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['close'],
            mode='lines',
            name='Close Price'
        ))

        # Add each SMA line
        for w in window_sizes:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df[f'sma_{w}'],
                mode='lines',
                name=f'SMA {w}'
            ))

        # --- Chart layout ---
        stock_name = df['name'].iloc[0] if 'name' in df.columns and not df['name'].empty else "Stock"
        fig.update_layout(
            title=f"Stock Price with SMAs for {stock_name}",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
        )

        return fig

    except Exception as e:
        print(f"Error generating SMA plot: {e}")
        return None


# --- plot runs using plotly ---
def plot_runs(runs_df, prices, min_length=4):
    """
    Creates a Plotly figure highlighting price runs (upward/downward trends).

    Parameters:
        runs_df (pd.DataFrame): DataFrame detailing the price runs.
        prices (pd.DataFrame): The original price data containing 'date' and 'close'.
        min_length (int): The minimum length of a run to be highlighted.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure object.
    """
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
            name=f"{run['direction']} Run (Length {run['length']})",
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
    )
    
    return fig

def plot_daily_returns_plotly(data, stock_name = "Stock"):
    """
    Creates an interactive bar chart showing daily returns for a given stock.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing stock data.
        stock_name (str): The name of the stock to visualize.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure object.
    """
    try:
        # Get filtered data with daily returns
        filtered = calculate_daily_returns(data)

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
                          template="plotly_white",
                          )
        
        return fig
    except Exception as e:
        print(f"Error generating Daily Returns plot: {e}")
        return None

def plot_max_profit_segments(data, stock_name = "Stock"):
    """
    Creates an interactive line chart showing stock price trend and annotates total maximum profit.
    """
    if 'close' not in data.columns:
        raise ValueError("'close' column not found in dataframe")
    
    prices = data['close'].reset_index(drop=True)
    
    # Calculate max profit (as currently implemented in metrics)
    total_profit = calculate_max_profit(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=prices, mode='lines', name='Price'))
    fig.update_layout(title=f"{stock_name} Max Profit Segments — Total Profit: ${total_profit:.2f}",
                      xaxis_title="Date", yaxis_title="Price ($)",
                      template="plotly_white",
                      )
    
    return fig


# Prediction Visualization 

def predicted_plot(historical_data, forecast_dates, forecast_values):
    """
    Generates a Plotly figure object plotting the historical stock prices 
    and the future forecast trend.
    """
    fig = go.Figure()
    
    # 1. Add the historical data line 
    fig.add_trace(go.Scatter(
        x=historical_data['date'],
        y=historical_data['close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1f77b4', width=2)
    ))

    # 2. Add the forecast trend line
    if forecast_dates and forecast_values:
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='Forecast Trend',
            line=dict(color='#ff7f0e', dash='dash', width=2)
        ))

        # 3. Highlight the next day prediction
        first_date = forecast_dates[0]
        first_value = forecast_values[0]

        fig.add_trace(go.Scatter(
            x=[first_date],
            y=[first_value],
            mode='markers',
            name='Next Day Prediction',
            marker=dict(
                color='#ff7f0e',
                size=16,
                symbol='star',
                line=dict(color='black', width=1)
            ),
            hovertemplate=f"<b>Next Day Forecast</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>"
        ))

        # Add annotation
        fig.add_annotation(
            x=first_date,
            y=first_value,
            text=f"<b>Next Day</b><br>${first_value:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=-20,
            ay=-60,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="rgba(255, 255, 255, 0.85)",
            opacity=0.8
        )

    # 4. Update layout for clarity
    fig.update_layout(
        title=f'Historical Data and Future Forecast',
        xaxis_title='Date',
        yaxis_title='Close Price ($)',
        template='plotly_white',
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01
        ),
    )

    return fig
