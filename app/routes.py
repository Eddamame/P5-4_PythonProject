from flask import (
    Blueprint, render_template, request, redirect,
    url_for, session, flash, current_app
)
from datetime import datetime, timedelta
import json
import pandas as pd
import plotly.io as pio

# 1. DEFINE THE BLUEPRINT OBJECT
main_bp = Blueprint('main', __name__)

from app.modules.data_fetcher import get_hist_data
from app.modules.data_handler import api_data_handler, handle_backup_csv, store_clean_data, retrieve_clean_data
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

from app.modules import validation  # Import the validation module


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

        # Store in session and proceed directly to metrics route for data fetch
        session['ticker'] = ticker
        session['period'] = period

        # Clear any previous analysis data
        # clear the session key pointing to the cache
        session.pop('data_cache_key', None)
        session.pop('selected_methods', None)
        session.pop('sma_window', None)
        session.pop('prediction_window', None)
        session.pop('run_length', None) # Clear run length setting

        return redirect(url_for('main.metrics'))


@main_bp.route('/metrics', methods=['GET', 'POST'])
def metrics():
    """
    Metrics route: Select analysis methods and fetch/clean data.
    Implements API fetch with fallback to backup data.
    """
    # Check if user has ticker/period in session
    if 'ticker' not in session or 'period' not in session:
        return redirect(url_for('main.index'))

    # Retrieve current ticker/period for rendering or data fetching
    ticker = session['ticker']
    period = session['period']
    
    # --- GET Request Handling ---
    if request.method == 'GET':
        # Render the metrics selection page
        return render_template('metrics.html',
                               ticker=ticker,
                               period=period,
                               error=None)

    # --- POST Request Handling (Data Fetching and Selection) ---
    if request.method == 'POST':
        
        selected_methods = []
        
        # Predictive Model
        if request.form.get('predictive_model'):
            selected_methods.append('predictive_model')
            prediction_window = request.form.get('prediction_window')
            if not prediction_window:
                return render_template('metrics.html', ticker=ticker, period=period, error="Please select a window size for Prediction before continuing.")
            session['prediction_window'] = int(prediction_window)
        else:
            session.pop('prediction_window', None)

        # Simple Moving Average (SMA)
        if request.form.get('sma'):
            selected_methods.append('sma')
            sma_window = request.form.get('sma_window')
            if not sma_window:
                return render_template('metrics.html', ticker=ticker, period=period, error="Please select a window size for SMA before continuing.")
            session['sma_window'] = sma_window
        else:
            session.pop('sma_window', None)

        # Daily Returns
        if request.form.get('daily_returns'):
            selected_methods.append('daily_returns')

        # Price Runs Analysis
        if request.form.get('runs'):
            selected_methods.append('runs')
            run_length = request.form.get('run_length') # Get run_length
            if not run_length: # Validate run_length
                return render_template('metrics.html', ticker=ticker, period=period, error="Please select a minimum run length before continuing.")
            session['run_length'] = int(run_length) # Store run_length for metrics display
        else:
            session.pop('run_length', None) # Clear if not selected

        # Max Profit
        if request.form.get('max_profit'):
            selected_methods.append('max_profit')

        if not selected_methods:
            return render_template('metrics.html', ticker=ticker, period=period, error="Please select at least one analysis method.")
        
        # --- Data Fetching Logic with Fallback ---
        try:
            # Use the original ticker for the API fetch, stripping the (BACKUP) tag if present
            current_ticker = ticker.replace(' (BACKUP)', '').strip()
            
            try:
                # Attempt 1: Fetch data from yfinance API
                raw_data = get_hist_data(current_ticker, period)

                if raw_data is None or raw_data.empty:
                    # Treat empty API response as a soft failure, forcing fallback
                    raise ValueError("API returned no data.")

                # Clean the API data
                clean_data = api_data_handler(raw_data, ticker=current_ticker)

            except Exception as api_e:
                # If API fails, use backup CSV handler
                current_app.logger.error(f"API data fetch failed for {current_ticker}: {api_e}. Falling back to backup CSV handler.")

                # Attempt 2: Load and Process Backup Data using the dedicated handler
                clean_data = handle_backup_csv(current_ticker, period)

                # Update session to indicate backup use and notify user
                session['ticker'] = f"{current_ticker} (BACKUP)"
                flash(f"Warning: Failed to fetch live data for {current_ticker}. Using backup historical dataset instead.", 'warning')


            if clean_data is None or clean_data.empty:
                # This catches if both API and backup failed or returned empty data
                raise ValueError("Both API fetch/clean and backup data processing failed.")

            # Store clean data using the in-memory cache and save the small key to the session
            data_cache_key = store_clean_data(clean_data)
            session['data_cache_key'] = data_cache_key
            
            # Store selections in session (small data is fine here)
            session['selected_methods'] = selected_methods

            return redirect(url_for('main.results'))

        except Exception as e:
            # Handle data fetching/cleaning errors
            error_msg = f"Error processing data: {str(e)}. Please try again."
            current_app.logger.error(f"Error processing data: {e}")
            
            # Use the (potentially updated) ticker for rendering the error page
            return render_template('metrics.html',
                                   ticker=session['ticker'],
                                   period=session['period'],
                                   error=error_msg)


@main_bp.route('/results')
def results():
    """
    Results route: Generate analyses and display dashboard
    """
    # 1. Check for required keys and retrieve data from cache
    required_keys = ['ticker', 'period', 'data_cache_key', 'selected_methods']
    for key in required_keys:
        if key not in session:
            return redirect(url_for('main.index'))

    try:
        # Retrieve small data from session
        ticker = session['ticker']
        period = session['period']
        selected_methods = session['selected_methods']

        # Retrieve DataFrame from the cache (and remove the key from session immediately)
        data_cache_key = session.pop('data_cache_key', None)

        if not data_cache_key:
            raise ValueError("Analysis data key not found in session. Please restart analysis.")
            
        clean_data = retrieve_clean_data(data_cache_key)
        
        if clean_data is None or clean_data.empty:
            raise ValueError("Analysis data not found in cache or is empty. Please restart the analysis.")

       
        

        # Initialize results dictionary
        analysis_results = {
            'ticker': ticker,
            'period': period,
            'plots': {},
            'metrics': {}
        }

        # --- Run analyses and generate plots ---
        
        # 1. Predictive Model (Forecasting)
        if 'predictive_model' in selected_methods:
            try:
                # Get the number of days to forecast from the session, defaulting to 10
                window_size = session.get('prediction_window', 10) 
                
                # forecast_prices now returns the plot HTML string directly, or None
                prediction_plot_html = forecast_prices(clean_data, 'close', window_size)
                
                if prediction_plot_html:
                    # Store the raw HTML string 
                    analysis_results['plots']['prediction'] = prediction_plot_html
                else:
                    # If None, the template will display 'Prediction plot unavailable.'
                    analysis_results['plots']['prediction'] = None
            except Exception as e:
                current_app.logger.error(f"Prediction model error: {e}")
                analysis_results['plots']['prediction'] = None
                analysis_results['metrics']['prediction_error'] = str(e)


        # 2. SMA (Simple Moving Average)
        if 'sma' in selected_methods:
            try:
                sma_input = session.get('sma_window', '20')
                if isinstance(sma_input, str):
                    window_sizes = [int(w.strip()) for w in sma_input.split(',') if w.strip().isdigit()]
                elif isinstance(sma_input, int):
                    window_sizes = [sma_input]
                else:
                    window_sizes = [20]

                fig = plot_price_and_sma(clean_data, window_sizes)
                if fig:
                    analysis_results['plots']['sma'] = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
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
                returns_data = calculate_daily_returns(clean_data)
                fig = plot_daily_returns_plotly(returns_data, ticker)
                if fig:
                    analysis_results['plots']['daily_returns'] = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

                    if not returns_data.empty and 'Daily_Return' in returns_data.columns:
                        analysis_results['metrics']['avg_daily_return'] = f"{returns_data['Daily_Return'].mean():.4f}"
                        analysis_results['metrics']['return_volatility'] = f"{returns_data['Daily_Return'].std():.4f}"
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
                fig = plot_max_profit_segments(clean_data, ticker)
                if fig:
                    analysis_results['plots']['max_profit'] = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
                else:
                    analysis_results['plots']['max_profit'] = None

                analysis_results['metrics']['total_max_profit'] = calculate_max_profit(clean_data)
            except Exception as e:
                current_app.logger.error(f"Max profit error: {e}")
                analysis_results['plots']['max_profit'] = None
                analysis_results['metrics']['max_profit_error'] = str(e)

        # 5. Runs Analysis
        if 'runs' in selected_methods:
            try:
                # Retrieve the run length from session. Default to 4 days, matching the plot_runs default.
                min_length_for_plot = session.get('run_length', 4)
                
                # calculate_runs remains unchanged (it calculates ALL runs)
                runs_data = calculate_runs(clean_data)
                runs_df = runs_data[0]
                prices = runs_data[2]

                # Pass the user-selected length to plot_runs for visual filtering/highlighting
                fig = plot_runs(runs_df, prices, min_length=min_length_for_plot)
                
                if fig:
                    analysis_results['plots']['runs'] = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
                else:
                    analysis_results['plots']['runs'] = None

                if not runs_df.empty:
                    analysis_results['metrics']['total_runs'] = len(runs_df)
                    analysis_results['metrics']['avg_run_length'] = f"{runs_df['length'].mean():.2f}"
                    analysis_results['metrics']['longest_run'] = runs_df['length'].max()
                else:
                    analysis_results['metrics']['total_runs'] = 0
                    analysis_results['metrics']['avg_run_length'] = 0.0
                    analysis_results['metrics']['longest_run'] = 0
                
                # Add the setting the user selected for display
                analysis_results['metrics']['min_run_length_setting'] = min_length_for_plot

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

@main_bp.route('/validate')
def validate():
    """
    Validation route: Run all validation tests and display results
    """
    try:
        # Import here to avoid circular imports
        from app.modules.validation import run_all_validations, validation_log
        
        # Clear previous validation log
        validation_log.clear()
        
        # Run all validation tests
        validation_results = run_all_validations()
        
        # Calculate summary statistics
        total_passed = 0
        total_tests = 0
        category_results = []
        
        for category, result_data in validation_results.items():
            passed = result_data['passed']
            total = result_data['total']
            tests = result_data['tests']
            
            total_passed += passed
            total_tests += total
            
            percentage = (passed / total * 100) if total > 0 else 0
            status = 'success' if passed == total else 'warning' if passed > 0 else 'danger'
            
            category_results.append({
                'name': category.replace('_', ' ').title(),
                'passed': passed,
                'total': total,
                'percentage': percentage,
                'status': status,
                'tests': tests  # Include individual test details
            })
        
        overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        if total_passed == total_tests:
            overall_status = 'success'
            overall_message = 'All tests passed! Your system is ready for integration.'
        elif overall_percentage >= 80:
            overall_status = 'warning'
            overall_message = 'Most tests passed. Minor issues to address.'
        elif overall_percentage >= 60:
            overall_status = 'warning'
            overall_message = 'Partial success. Some components need debugging.'
        else:
            overall_status = 'danger'
            overall_message = 'Significant issues detected. Please review failed tests.'
        
        summary = {
            'total_passed': total_passed,
            'total_tests': total_tests,
            'percentage': overall_percentage,
            'status': overall_status,
            'message': overall_message
        }
        
        # Get validation log for detailed output
        log_output = '\n'.join(validation_log)
        
        
        return render_template('validate.html',
                             summary=summary,
                             categories=category_results,
                             log_output=log_output)
    
    except Exception as e:
        current_app.logger.error(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        flash(f"Error running validation tests: {str(e)}", 'error')
        return redirect(url_for('main.index'))

@main_bp.route('/reset')
def reset():
    """
    Route to clear session and start over (used by 'Inspect New Ticker' button)
    """
    session.clear()
    return redirect(url_for('main.index'))


@main_bp.errorhandler(404)
def page_not_found(e):
    """
    Handle 404 errors
    """
    return render_template('404.html'), 404


@main_bp.errorhandler(500)
def internal_server_error(e):
    """
    Handle 500 errors
    """
    return render_template('500.html'), 500
