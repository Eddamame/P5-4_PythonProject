# Use this for your visualization functions 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Union
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

def plot_daily_returns_plotly(data, stock_name = str):

    """
    Creates an interactive bar chart showing daily returns for a given stock.
    Parameters:
        data (pd.DataFrame): DataFrame containing stock data from api_data_handler.
        stock_name (str): The name of the stock to visualize.
        start_date (str, optional): Start date filter in 'YYYY-MM-DD' format.
        end_date (str, optional): End date filter in 'YYYY-MM-DD' format.
    """
    # Check if required columns exist
    if 'date' not in data.columns:
        raise ValueError("'date' column not found in dataframe")
    if 'close' not in data.columns:
        raise ValueError("'close' column not found in dataframe")
    
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
                      template="plotly_white")
    fig.show()

def plot_max_profit_segments(data):
    """
    Creates an interactive line chart showing stock price trend and annotates total maximum profit.
    Parameters:
        data (pd.DataFrame): DataFrame containing stock data from api_data_handler.
        stock_name (str): The name of the stock to visualize.
        start_date (str, optional): Start date filter in 'YYYY-MM-DD' format.
        end_date (str, optional): End date filter in 'YYYY-MM-DD' format.

    Plot the stock price series and highlight all buy–sell segments
    that contribute to the maximum profit (Valley–Peak strategy),
    while calling calculate_max_profit to display the total.
    """
    
    if 'close' not in data.columns:
        raise ValueError("'close' column not found in dataframe")
    
    prices = data['close'].reset_index(drop=True)
    total_profit = calculate_max_profit(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=prices, mode='lines', name='Price'))
    fig.update_layout(title=f"Max Profit Segments — Total Profit: {total_profit}",
                      xaxis_title="Date", yaxis_title="Price ($)",
                      template="plotly_white")
    fig.show()


#  Prediction Visualization
def validation_plot(test_dates, actual_prices, predicted_prices):
    """
    Creates a simple line graph comparing actual and predicted prices over time.

    Parameters:
        test_dates (pd.Series): The dates corresponding to the test set.
        actual_prices (np.array): The actual price values from the test set.
        predicted_prices (np.array): The values predicted by the model.

    Output:
        Line Graph: Shows actual vs predicted prices
    """
    # Prep dataframe
    df = pd.DataFrame({
        'date': test_dates,
        'actual': actual_prices,
        'predicted': predicted_prices
    })
    
    # Sort by date
    df = df.sort_values(by='date')
    
    fig = go.Figure()

    # Add actual prices (historical data)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['actual'],
        mode='lines+markers',
        name='Actual Price',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))

    # Add predicted prices
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['predicted'],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    # Format layout
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

    Output:
        Table: Shows actual price, predicted price, difference and percentage difference.
    """
    # Dataframe prep
    df = pd.DataFrame({
        'Date': test_dates,
        'Actual': actual_prices,
        'Predicted': predicted_prices
    })
    df = df.sort_values(by='Date')
    
    # Differences
    df['Difference'] = df['Actual'] - df['Predicted']
    df['Difference %'] = (df['Difference'] / df['Actual'] * 100).round(2)
    
    # Format to 2 decimal places for $
    df['Actual'] = df['Actual'].round(2)
    df['Predicted'] = df['Predicted'].round(2)
    df['Difference'] = df['Difference'].round(2)
    
    # Figure formatting
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

def predicted_plot(historical_data, forecast_dates, forecast_values):
    """
    Plots historical values and predicted values together

    Parameters:
        historical_data (pd.DataFrame): Original dataframe
        forecast_dates (list): List of dates for predicted values
        forecast_values (list): List of predicted values
    
    Output:
        Line Graph: Shows historical data and predicted prices, but highlights next day prediction
    """
    fig = go.Figure()

    # 1. Add the historical data line with a professional look
    fig.add_trace(go.Scatter(
        x=historical_data['date'],
        y=historical_data['close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1f77b4', width=2)  # A nice blue
    ))

    # 2. Add the forecast trend line (dashed, no markers for a cleaner look)
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines',
        name='Forecast Trend',
        line=dict(color='#ff7f0e', dash='dash', width=2)  # A vibrant orange
    ))

    # 3. Highlight the single most important prediction: the next day
    if forecast_dates and forecast_values:
        first_date = forecast_dates[0]
        first_value = forecast_values[0]

        # Add a distinct star marker for the first predicted day
        fig.add_trace(go.Scatter(
            x=[first_date],
            y=[first_value],
            mode='markers',
            name='Next Day Prediction',
            marker=dict(
                color='#ff7f0e',  # Match the forecast color
                size=16,
                symbol='star',
                line=dict(color='black', width=1)
            ),
            hovertemplate=f"<b>Next Day Forecast</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>"
        ))

        # Add a well-placed annotation that won't overlap
        fig.add_annotation(
            x=first_date,
            y=first_value,
            text=f"<b>Next Day</b><br>${first_value:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=-20,  # Horizontal offset of the text from the arrow
            ay=-60,  # Vertical offset of the text from the arrow
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="rgba(255, 255, 255, 0.85)", # Semi-transparent background
            opacity=0.8
        )

    # 4. Update layout for clarity
    fig.update_layout(
        title='Stock Price: Historical Data and Future Forecast',
        xaxis_title='Date',
        yaxis_title='Close Price ($)',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
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