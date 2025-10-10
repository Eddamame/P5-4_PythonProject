from flask import Flask, render_template, request, redirect, url_for, session, flash
from datetime import datetime, timedelta
import json

from app.modules.data_fetcher import get_hist_data
from app.modules.data_handler import api_data_handler
from app.modules.prediction import forecast_prices
from app.modules.visualization import (
    predicted_plot, 
    plot_price_and_sma,
    plot_daily_returns_plotly,
    plot_max_profit_segments, 
    plot_runs)
from app.modules.metrics import (
    calculate_sma,
    calculate_daily_returns,
    calculate_max_profit,
    calculate_runs
)

app = Flask(__name__)
app.secret_key = 'P5-4'  # Change this to a secure random key

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Index route: Collect and validate ticker + period
    """
    if request.method == 'GET':
        # Render the input form
        return render_template('index.html', error=None)
    
    if request.method == 'POST':
        # Get form data
        ticker = request.form.get('ticker', '').strip().upper()
        period = request.form.get('period', '')
        
        # Basic validation
        if not ticker or not period:
            return render_template('index.html', 
                                 error="Please fill in both ticker and period.")
        
        # Validate ticker by attempting to fetch data
        try:
            # Quick validation call
            test_data = get_hist_data(ticker, period)
            if test_data is None or len(test_data) == 0:
                raise ValueError("No data returned")
            
            # If valid, store in session
            session['ticker'] = ticker
            session['period'] = period
            
            # Clear any previous analysis data
            session.pop('clean_data', None)
            session.pop('selected_methods', None)
            session.pop('sma_window', None)
            
            return redirect(url_for('metrics'))
            
        except Exception as e:
            # Invalid ticker or API error
            error_msg = "Invalid ticker. Please try again with a valid stock symbol."
            return render_template('index.html', error=error_msg)


@app.route('/metrics', methods=['GET', 'POST'])
def metrics():
    """
    Metrics route: Select analysis methods and fetch/clean data
    """
    # Check if user has ticker/period in session
    if 'ticker' not in session or 'period' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'GET':
        # Render the metrics selection page
        return render_template('metrics.html', 
                             ticker=session['ticker'],
                             period=session['period'],
                             error=None)
    
    if request.method == 'POST':
        # Get selected analysis methods
        selected_methods = []
        
        # Check which methods were selected
        if request.form.get('predictive_model'):
            selected_methods.append('predictive_model')
            # Get prediction window size
            prediction_window = request.form.get('prediction_window')

            # Validate selection
            if not prediction_window:
                return render_template('metrics.html',
                                    ticker=session['ticker'],
                                    period=session['period'],
                                    error="Please select a window size for Prediction before continuing.")

            # Store in session
            session['prediction_window'] = int(prediction_window)
        
        if request.form.get('sma'):
            selected_methods.append('sma')
            # Get SMA window size
            sma_window = request.form.get('sma_window')
            
            # Validate SMA window selection
            if not sma_window:
                return render_template('metrics.html',
                                     ticker=session['ticker'],
                                     period=session['period'],
                                     error="Please select a window size for SMA before continuing.")
            
            # Store SMA window in session
            session['sma_window'] = int(sma_window)
        
        if request.form.get('daily_returns'):
            selected_methods.append('daily_returns')
        
        if request.form.get('runs'):
            selected_methods.append('runs')
        
        if request.form.get('max_profit'):
            selected_methods.append('max_profit')
        
        # Validate at least one method was selected
        if not selected_methods:
            return render_template('metrics.html',
                                 ticker=session['ticker'],
                                 period=session['period'],
                                 error="Please select at least one analysis method.")
        
        try:
            # Fetch and clean data
            ticker = session['ticker']
            period = session['period']
            
            # Get historical data
            raw_data = get_hist_data(ticker, period)
            
            if raw_data is None:
                raise ValueError("Failed to fetch data")
            
            # Clean the data
            clean_data = api_data_handler(raw_data)
            
            if clean_data is None or clean_data.empty:
                raise ValueError("Data cleaning failed")
            
            # Store clean data and selections in session
            # Convert DataFrame to JSON for session storage
            session['clean_data'] = clean_data.to_json()
            session['selected_methods'] = selected_methods
            
            return redirect(url_for('results'))
            
        except Exception as e:
            # Handle data fetching/cleaning errors
            error_msg = f"Error processing data: {str(e)}. Please try again."
            return render_template('metrics.html',
                                 ticker=session['ticker'],
                                 period=session['period'],
                                 error=error_msg)


@app.route('/results')
def results():
    """
    Results route: Generate analyses and display dashboard
    """
    # Validate session data exists
    required_keys = ['ticker', 'period', 'clean_data', 'selected_methods']
    for key in required_keys:
        if key not in session:
            return redirect(url_for('index'))
    
    try:
        # Retrieve data from session
        ticker = session['ticker']
        period = session['period']
        selected_methods = session['selected_methods']
        
        # Convert JSON back to DataFrame
        import pandas as pd
        clean_data = pd.read_json(session['clean_data'])
        
        # Initialize results dictionary
        analysis_results = {
            'ticker': ticker,
            'period': period,
            'plots': {},
            'metrics': {}
        }
        
        # Run analyses in fixed order (as per requirements)
        
        # 1. Predictive Model
        if 'predictive_model' in selected_methods:
            try:
                # Run Forecast. default value for prediction window: 10 days
                window_size = session.get('prediction_window', 10)

                # Generate only the chart
                plot_data = forecast_prices(clean_data, window_size)
                analysis_results['plots']['prediction'] = plot_data

            except Exception as e:
                analysis_results['plots']['prediction'] = None
                analysis_results['metrics']['prediction_error'] = str(e)

        
        # 2. SMA (Simple Moving Average)
        if 'sma' in selected_methods:
            try:
                window_size = session.get('sma_window', 20)
                
                # Calculate SMA
                sma_data = calculate_sma(clean_data, window_size)
                
                # Generate SMA plot
                plot_data = plot_price_and_sma(sma_data, window_size)
                analysis_results['plots']['sma'] = plot_data
                analysis_results['metrics']['sma_window'] = window_size
            except Exception as e:
                analysis_results['plots']['sma'] = None
                analysis_results['metrics']['sma_error'] = str(e)
        
        # 3. Daily Returns
        if 'daily_returns' in selected_methods:
            try:
                # Calculate daily returns
                returns_data = calculate_daily_returns(clean_data, ticker)
                
                # Generate returns plot
                plot_data = plot_daily_returns_plotly(returns_data)
                analysis_results['plots']['daily_returns'] = plot_data
                
                # Add summary statistics
                analysis_results['metrics']['avg_daily_return'] = returns_data['Daily_Return'].mean()
                analysis_results['metrics']['return_volatility'] = returns_data['Daily_Return'].std()

            except Exception as e:
                analysis_results['plots']['daily_returns'] = None
                analysis_results['metrics']['daily_returns_error'] = str(e)
        
        # 4. Max Profit
        if 'max_profit' in selected_methods:
            try:
                # Plot data and return max profit
                plot_data = plot_max_profit_segments(clean_data, ticker)
                analysis_results['plots']['max_profit'] = plot_data
                
            except Exception as e:
                analysis_results['plots']['max_profit'] = None
                analysis_results['metrics']['max_profit_error'] = str(e)
        
        # 5. Runs Analysis
        if 'runs' in selected_methods:
            try:
                # Unpack the tuple returned by calculate_runs
                runs_df, direction, df = calculate_runs(clean_data)

                # Generate runs plot using the DataFrame
                plot_data = plot_runs(runs_df)
                analysis_results['plots']['runs'] = plot_data

                # Add runs statistics derived from runs_df
                analysis_results['metrics']['total_runs'] = len(runs_df)
                analysis_results['metrics']['avg_run_length'] = runs_df['length'].mean()
                analysis_results['metrics']['longest_run'] = runs_df['length'].max()

            except Exception as e:
                analysis_results['plots']['runs'] = None
                analysis_results['metrics']['runs_error'] = str(e)
        
        # Render results page with all analyses
        return render_template('results.html', 
                             results=analysis_results,
                             selected_methods=selected_methods)
        
    except Exception as e:
        # If any critical error occurs, redirect to index with error message
        flash(f"Error generating results: {str(e)}", 'error')
        return redirect(url_for('index'))


@app.route('/reset')
def reset():
    """
    Optional route to clear session and start over
    """
    session.clear()
    return redirect(url_for('index'))


@app.errorhandler(404)
def page_not_found(e):
    """
    Handle 404 errors
    """
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    """
    Handle 500 errors
    """
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(debug=True)