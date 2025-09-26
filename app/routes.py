# Handles all user interaction / requests

# In routes.py
from flask import render_template


@app.route('/')
def select_stock():
    # hardcode TOP 10 Stocks in HTML
    top_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "TSLA", "BRK-B", "JPM"]
    return render_template('index.html', stocks=top_stocks)

@app.route('/analyze', methods=['POST'])
def analyze():
    
    pass 

if __name__ == '__main__':
    app.run(debug=True)