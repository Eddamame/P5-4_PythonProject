import numpy as np
import pandas as pd

def fit_coefficients(X, y):
    """Fit regression coefficients using the Normal Equation."""
    # Adding intercept term
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    # Calculate coefficients using the formula: (X'X)^(-1) X'y
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
    return coefficients

def predict(X, coefficients):
    """Predict values using the coefficients."""
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return X @ coefficients

def multiple_linear_regression(data, target_column, days_ahead):
    """
    Perform multiple linear regression and predict the next 'days_ahead' values.
    
    Args:
    - data (pd.DataFrame): Dataframe containing the stock data.
    - target_column (str): The column name to predict (e.g., 'close').
    - days_ahead (int): Number of days to predict in the future.

    Returns:
    - future_predictions (pd.DataFrame): Dataframe with future dates and predicted values.
    """
    # Separate features (X) and target (y)
    features = data.drop(columns=['date', 'name', target_column]).values
    target = data[target_column].values

    # Fit coefficients using training data
    coefficients = fit_coefficients(features, target)

    # Create future predictions based on the last row of features
    last_row = features[-1]
    future_features = np.tile(last_row, (days_ahead, 1))
    future_predictions = predict(future_features, coefficients)

    # Generate future dates
    last_date = pd.to_datetime(data['date'].iloc[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_ahead)

    # Combine future predictions into a DataFrame
    result = pd.DataFrame({'date': future_dates, target_column: future_predictions})
    return result