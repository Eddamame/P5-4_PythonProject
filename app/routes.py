from flask import (
    Blueprint, render_template, request, redirect, 
    url_for, session, flash, current_app
)
from datetime import datetime, timedelta
import json
import pandas as pd
import plotly.io as pio # <-- ADDED: For Plotly figure serialization

# 1. DEFINE THE BLUEPRINT OBJECT
# This object records all your routes.
main_bp = Blueprint('main', __name__)

from app.modules.data_fetcher import get_hist_data
from app.modules.data_handler import api_data_handler
from app.modules.prediction import forecast_prices
from app.modules.visualization import (
    predicted_plot, 
    plot_price_and_sma,
    plot_daily_returns_plotly,
    plot_max_profit_segments, 
    plot_runs
)
from app.modules.metrics import (
    calculate_sma,
    calculate_daily_returns,
    calculate_max_profit,
    calculate_runs
)


@main_bp.route('/', methods=['GET', 'POST'])
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
            
            # Note: We use the Blueprint name ('main') to reference the endpoint
            return redirect(url_for('main.metrics')) 
            
        except Exception as e:
            # Invalid ticker or API error
            error_msg = f"Invalid ticker or data error: {str(e)}. Please try again."
            # We can log the error if needed
            current_app.logger.error(f"Ticker validation failed for {ticker}: {e}")
            return render_template('index.html', error=error_msg)


@main_bp.route('/metrics', methods=['GET', 'POST'])
def metrics():
    """
    Metrics route: Select analysis methods and fetch/clean data
    """
    # Check if user has ticker/period in session
    if 'ticker' not in session or 'period' not in session:
        return redirect(url_for('main.index'))
    
    # ... (rest of the metrics function remains the same, but imports are removed) ...
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
            # Note: Assuming this allows multiple selections (e.g., "10,20,50") if the form supports it.
            # Otherwise, store as a single integer. For simplicity, we'll store the raw string/int for now.
            session['sma_window'] = sma_window
        
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
            session['clean_data'] = clean_data.to_json(date_format='iso', orient='split') # Using a standard format
            session['selected_methods'] = selected_methods
            
            return redirect(url_for('main.results'))
            
        except Exception as e:
            # Handle data fetching/cleaning errors
            error_msg = f"Error processing data: {str(e)}. Please try again."
            current_app.logger.error(f"Error processing data: {e}")
            return render_template('metrics.html',
                                   ticker=session['ticker'],
                                   period=session['period'],
                                   error=error_msg)


@main_bp.route('/results')
def results():
    """
    Results route: Generate analyses and display dashboard
    """
    # Validate session data exists
    required_keys = ['ticker', 'period', 'clean_data', 'selected_methods']
    for key in required_keys:
        if key not in session:
            return redirect(url_for('main.index'))
    
    try:
        # Retrieve data from session
        ticker = session['ticker']
        period = session['period']
        selected_methods = session['selected_methods']
        
        # Convert JSON back to DataFrame
        # Ensure we read the date column correctly
        clean_data = pd.read_json(session['clean_data'], orient='split')
        clean_data['date'] = pd.to_datetime(clean_data['date'])
        
        # Initialize results dictionary
        analysis_results = {
            'ticker': ticker,
            'period': period,
            'plots': {},
            'metrics': {}
        }
        
        # Run analyses in fixed order (as per requirements)
        
        # 1. Predictive Model (Forecasting)
        if 'predictive_model' in selected_methods:
            try:
                # Run Forecast. default value for prediction window: 10 days
                window_size = session.get('prediction_window', 10)

                # NOTE: forecast_prices needs 3 arguments: data, target_column, n_days
                # Assuming 'close' is the target column based on prediction.py context
                forecast_data = forecast_prices(clean_data, 'close', window_size)
                
                # forecast_prices in prediction.py now returns (forecast_dates, forecast_values)
                # We use these to generate the plot here.
                if forecast_data:
                    forecast_dates, forecast_values = forecast_data
                    # Generate the plot
                    fig = predicted_plot(clean_data, forecast_dates, forecast_values)
                    # Convert Plotly figure to JSON string for passing to template
                    analysis_results['plots']['prediction'] = pio.to_json(fig)
                else:
                    analysis_results['plots']['prediction'] = None

            except Exception as e:
                current_app.logger.error(f"Prediction model error: {e}")
                analysis_results['plots']['prediction'] = None
                analysis_results['metrics']['prediction_error'] = str(e)

        
        # 2. SMA (Simple Moving Average)
        if 'sma' in selected_methods:
            try:
                # The window size can be a single int or a comma-separated string of ints.
                sma_input = session.get('sma_window', '20')
                if isinstance(sma_input, str):
                    window_sizes = [int(w.strip()) for w in sma_input.split(',') if w.strip().isdigit()]
                elif isinstance(sma_input, int):
                    window_sizes = [sma_input]
                else:
                    window_sizes = [20] # Default fallback

                # Generate SMA plot (plot_price_and_sma handles calculation internally)
                fig = plot_price_and_sma(clean_data, window_sizes)
                
                if fig:
                    # Convert Plotly figure to JSON string
                    analysis_results['plots']['sma'] = pio.to_json(fig)
                    analysis_results['metrics']['sma_window'] = ", ".join(map(str, window_sizes))
                else:
                     analysis_results['plots']['sma'] = None
            except Exception as e:
                current_app.logger.error(f"SMA error: {e}")
                analysis_results['plots']['sma'] = None
                analysis_results['metrics']['sma_error'] = str(e)
        
        # 3. Daily Returns
        if 'daily_returns' in selected_methods:
            try:
                # Calculate daily returns
                returns_data = calculate_daily_returns(clean_data) # Removed ticker argument, as calculate_daily_returns doesn't use it.
                
                # Generate returns plot
                fig = plot_daily_returns_plotly(returns_data, ticker)
                
                if fig:
                    analysis_results['plots']['daily_returns'] = pio.to_json(fig)
                    
                    # Add summary statistics
                    # Avoid mean/std on empty set
                    if not returns_data.empty and 'Daily_Return' in returns_data.columns:
                        analysis_results['metrics']['avg_daily_return'] = returns_data['Daily_Return'].mean()
                        analysis_results['metrics']['return_volatility'] = returns_data['Daily_Return'].std()
                    else:
                        analysis_results['metrics']['avg_daily_return'] = 0.0
                        analysis_results['metrics']['return_volatility'] = 0.0

            except Exception as e:
                current_app.logger.error(f"Daily returns error: {e}")
                analysis_results['plots']['daily_returns'] = None
                analysis_results['metrics']['daily_returns_error'] = str(e)
        
        # 4. Max Profit
        if 'max_profit' in selected_methods:
            try:
                # Plot data and return max profit
                fig = plot_max_profit_segments(clean_data, ticker)
                
                if fig:
                    analysis_results['plots']['max_profit'] = pio.to_json(fig)
                else:
                    analysis_results['plots']['max_profit'] = None
                    
                # Add total profit metric directly
                analysis_results['metrics']['total_max_profit'] = calculate_max_profit(clean_data)
                
            except Exception as e:
                current_app.logger.error(f"Max profit error: {e}")
                analysis_results['plots']['max_profit'] = None
                analysis_results['metrics']['max_profit_error'] = str(e)
        
        # 5. Runs Analysis
        if 'runs' in selected_methods:
            try:
                # Unpack the tuple returned by calculate_runs
                # calculate_runs returns (runs_df, direction (now removed from metric), prices)
                runs_data = calculate_runs(clean_data)
                runs_df = runs_data[0]
                prices = runs_data[2] # Use the original df or df from calculate_runs (index 2)

                # Generate runs plot using the DataFrame (assuming default min_length=4 in visualization)
                fig = plot_runs(runs_df, prices)
                
                if fig:
                    analysis_results['plots']['runs'] = pio.to_json(fig)
                else:
                    analysis_results['plots']['runs'] = None

                # Add runs statistics derived from runs_df
                if not runs_df.empty:
                    analysis_results['metrics']['total_runs'] = len(runs_df)
                    analysis_results['metrics']['avg_run_length'] = runs_df['length'].mean()
                    analysis_results['metrics']['longest_run'] = runs_df['length'].max()
                else:
                    analysis_results['metrics']['total_runs'] = 0
                    analysis_results['metrics']['avg_run_length'] = 0.0
                    analysis_results['metrics']['longest_run'] = 0

            except Exception as e:
                current_app.logger.error(f"Runs analysis error: {e}")
                analysis_results['plots']['runs'] = None
                analysis_results['metrics']['runs_error'] = str(e)
        
        # Render results page with all analyses
        return render_template('results.html', 
                               results=analysis_results,
                               selected_methods=selected_methods)
        
    except Exception as e:
        # If any critical error occurs, redirect to index with error message
        current_app.logger.error(f"Critical error generating results: {e}")
        flash(f"Critical error generating results: {str(e)}", 'error')
        return redirect(url_for('main.index'))


@main_bp.route('/reset')
def reset():
    """
    Optional route to clear session and start over
    """
    session.clear()
    return redirect(url_for('main.index'))


@main_bp.errorhandler(404)
def page_not_found(e):
    """
    Handle 404 errors (uses main_bp.errorhandler now)
    """
    return render_template('404.html'), 404


@main_bp.errorhandler(500)
def internal_server_error(e):
    """
    Handle 500 errors (uses main_bp.errorhandler now)
    """
    return render_template('500.html'), 500
