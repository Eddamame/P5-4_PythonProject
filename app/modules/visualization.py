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

def validation_plot(test_dates, actual_prices, predicted_prices):
    """
    Creates a simple line graph comparing actual and predicted prices over time.

    Parameters:
        test_dates (pd.Series): The dates corresponding to the test set.
        actual_prices (np.array): The actual price values from the test set.
        predicted_prices (np.array): The values predicted by the model.
    """
    # Convert everything to pandas Series/DataFrame for easier sorting
    df = pd.DataFrame({
        'date': test_dates,
        'actual': actual_prices,
        'predicted': predicted_prices
    })
    
    # Sort by date to ensure chronological order
    df = df.sort_values(by='date')
    
    # Create a new figure
    fig = go.Figure()

    # Add the RED line for ACTUAL prices with markers
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['actual'],
        mode='lines+markers',
        name='Actual Price',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))

    # Add the BLUE line for PREDICTED prices with markers
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['predicted'],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    # Simple, clean layout
    fig.update_layout(
        title='Actual vs. Predicted Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template="plotly_white",
        width=900,
        height=500
    )

    fig.show()

def validation_table(test_dates, actual_prices, predicted_prices):
    """
    Creates and displays a table comparing actual vs. predicted prices with their difference.
    
    Parameters:
        test_dates (pd.Series): The dates corresponding to the test set.
        actual_prices (np.array): The actual price values from the test set.
        predicted_prices (np.array): The values predicted by the model.
        
    Returns:
        pd.DataFrame: DataFrame containing the comparison data.
    """
    # Create and sort the DataFrame
    df = pd.DataFrame({
        'Date': test_dates,
        'Actual': actual_prices,
        'Predicted': predicted_prices
    })
    df = df.sort_values(by='Date')
    
    # Calculate differences
    df['Difference'] = df['Actual'] - df['Predicted']
    df['Difference %'] = (df['Difference'] / df['Actual'] * 100).round(2)
    
    # Format the numeric columns to 2 decimal places
    df['Actual'] = df['Actual'].round(2)
    df['Predicted'] = df['Predicted'].round(2)
    df['Difference'] = df['Difference'].round(2)
    
    # Create a table visualization
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Date', 'Actual Price', 'Predicted Price', 'Difference', 'Difference (%)'],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[
                df['Date'].dt.strftime('%Y-%m-%d'),
                df['Actual'],
                df['Predicted'],
                df['Difference'],
                df['Difference %'].apply(lambda x: f"{x:+.2f}%")
            ],
            fill_color=[
                'white',
                'white',
                'white',
                [
                    'lightgreen' if val >= 0 else 'lightpink' 
                    for val in df['Difference']
                ],
                [
                    'lightgreen' if val >= 0 else 'lightpink' 
                    for val in df['Difference %']
                ]
            ],
            align='right',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title='Actual vs. Predicted Prices Comparison',
        width=800
    )
    
    fig.show()
    
    # Return the DataFrame in case you want to use it for further analysis
    return df

def predicted_plot(historical_data, forecast_dates, forecast_values):
    """
    Plots the historical closing prices and overlays the future forecasted prices.

    Parameters:
        historical_data (pd.DataFrame): The full dataframe with 'date' and 'close' columns.
        forecast_dates (pd.DatetimeIndex): The future dates for the forecast.
        forecast_values (list): The predicted values for the future dates.
    """
    fig = go.Figure()

    # Add the historical data line
    fig.add_trace(go.Scatter(
        x=historical_data['date'],
        y=historical_data['close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='blue')
    ))

    # Add the forecast line
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines+markers',
        name='Forecasted Price',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title='Historical Prices with Future Forecast',
        xaxis_title='Date',
        yaxis_title='Close Price',
        showlegend=True,
        width=1000,
        height=600
    )

    fig.show()


# --- Test Block --- 
# data = get_hist_data('PLTR', '12mo')
# df = api_data_handler(data)
# print(df)
# runs_df, direction, prices = calculate_runs(df)
## Testing plot runs 
# my_plot = plot_runs(runs_df, prices, 6)
# if my_plot is not None:
#     my_plot.show()