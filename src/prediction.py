import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def fit_coefficients(X, y):
    """
    Fit regression coefficients using the Normal Equation with pseudoinverse.
    """
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))  # Add intercept
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta


def predict(X, theta):
    """
    Predict values given features X and coefficients theta.
    """
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))
    return X @ theta


def train_test_split_and_fit(data, target_column, test_size=0.2):
    """
    Split data into training and testing sets, fit the model, and return coefficients.

    Args:
    - data: DataFrame containing the features and target.
    - target_column: The target column name (e.g., 'close').
    - test_size: Fraction of data to use as the test set.

    Returns:
    - coefficients: Fitted coefficients for the multiple linear regression model.
    - X_test: Test dataset features.
    - y_test: Test dataset actual values.
    """
    # Prepare features and target
    X = data.drop(columns=['date', 'name', target_column]).values
    y = data[target_column].values

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Fit coefficients using training data
    coefficients = fit_coefficients(X_train, y_train)

    return coefficients, X_test, y_test


def validate_and_plot(data, target_column, test_size=0.2):
    """
    Train, test, and plot actual vs. predicted values for the test dataset.

    Args:
    - data: DataFrame containing the features and target.
    - target_column: The target column name (e.g., 'close').
    - test_size: Fraction of data to use as the test set.
    """
    # Train and test the model
    coefficients, X_test, y_test = train_test_split_and_fit(data, target_column, test_size)

    # Extract corresponding test dates
    test_dates = data['date'].iloc[len(data) - len(y_test):].reset_index(drop=True)

    # Predict values for the test set
    y_pred = predict(X_test, coefficients)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-Squared: {r2:.4f}")

    # Plot actual vs. predicted values with improved visualization
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test, label="Actual Stock Prices", marker="o", linestyle="--", color="blue")
    plt.plot(test_dates, y_pred, label="Predicted Stock Prices", marker="x", linestyle="--", color="orange")
    plt.title(f"Actual vs. Predicted Stock Prices for Test Dataset\n(Target: {target_column})", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Stock Price", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def predict_next_day(data, target_column):
    """
    Predict the next day's value based on the most recent data.

    Args:
    - data: DataFrame containing the features and target.
    - target_column: The target column name (e.g., 'close').

    Returns:
    - next_day_prediction: Predicted value for the next day.
    """
    # Prepare features and target
    X = data.drop(columns=['date', 'name', target_column]).values
    y = data[target_column].values

    # Fit coefficients on the full dataset
    coefficients = fit_coefficients(X, y)

    # Use the last row of features for prediction
    last_features = X[-1].reshape(1, -1)
    next_day_prediction = predict(last_features, coefficients)[0]

    print(f"Predicted {target_column} for the next day: {next_day_prediction:.2f}")
    return next_day_prediction

def plot_actual_prices(data, target_column):
    """
    Plot actual stock prices over time, ensuring proper handling of dates and values.

    Args:
    - data: DataFrame containing the stock data.
    - target_column: The name of the column to plot (e.g., 'close').
    """
    # Ensure the data is sorted by date
    data = data.sort_values(by='date')

    # Optionally remove duplicates and outliers
    data = data.drop_duplicates(subset='date')
    Q1 = data[target_column].quantile(0.25)
    Q3 = data[target_column].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data[target_column] >= Q1 - 1.5 * IQR) & (data[target_column] <= Q3 + 1.5 * IQR)]

    # Plot the actual stock prices
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data[target_column], label="Actual Stock Prices", color="blue", marker="o", linestyle="-")
    plt.title(f"Actual Stock Prices Over Time\n(Target: {target_column})", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Stock Price", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()