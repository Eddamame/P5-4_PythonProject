# Handles all user interaction / requests

from flask import Blueprint, render_template, request, flash, redirect, url_for
from app.module.data_fetcher import fetch_historical_data

# Assuming you are using a Blueprint for organization
main = Blueprint('main', __name__) 

@main.route('/analyze', methods=['POST'])
def analyze_stock():
    ticker = request.form.get('ticker')
    period = request.form.get('period')
    
    # 1. (Placeholder) Call Validation Module
    # validation.validate_input(ticker, period) 
    
    # 2. Call the data fetcher
    data_response = fetch_historical_data(ticker, period)
    
    # 3. Handle the response
    if isinstance(data_response, dict) and data_response.get('status') == 'error':
        # Logged error detected! Display the user-friendly message from the fetcher.
        flash(data_response['message'], 'danger') # 'danger' is a category for styling
        
        # Redirect back to the input form page (index.html)
        return redirect(url_for('main.index')) 

    # 4. Success Case: Process the DataFrame
    stock_df = data_response
    
    # ... Continue to call data_handler, metrics, prediction modules
    # processed_data = data_handler.process(stock_df)
    
    # ... Render the results page
    return render_template('results.html', data=stock_df.to_html()) 

# Assuming a simple route for your main form page
@main.route('/')
def index():
    return render_template('index.html')