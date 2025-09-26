import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fit_coefficients(X, y):
    """Fit regression coefficients using the Normal Equation with pseudoinverse."""
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))  # add intercept
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

def predict(X, theta):
    """Predict values given features X and coefficients theta."""
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))
    return X.dot(theta)

# ------------------------------
# Validation (Train/Test split)
# ------------------------------
def validate_regression(df, target="close", train_ratio=0.7):
    """Train on train_ratio% of data, test on the rest."""
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    split_idx = int(len(df) * train_ratio)
    train_df = df[:split_idx]
    test_df = df[split_idx:]

    X_train = train_df.drop(columns=[target, "date", "Name"]).values
    y_train = train_df[target].values
    X_test = test_df.drop(columns=[target, "date", "Name"]).values
    y_test = test_df[target].values

    theta = fit_coefficients(X_train, y_train)
    y_pred = predict(X_test, theta)

    return theta, y_test, y_pred

# ------------------------------
# Forecast (Future N days)
# ------------------------------
def forecast_regression(df, target="close", days_ahead=10, show_graph=True):
    """
    Fit regression on full dataset and forecast 'days_ahead' into the future.

    Returns:
        table (DataFrame): Future dates with predicted values.
    """
    # Features and target
    X = df.drop(columns=[target, "date", "Name"]).values
    y = df[target].values

    # Train model
    theta = fit_coefficients(X, y)

    # Future feature assumption = repeat last row
    last_features = df.drop(columns=[target, "date", "Name"]).iloc[-1].values
    future_X = np.tile(last_features, (days_ahead, 1))

    # Predictions
    y_future = predict(future_X, theta)

    # Future dates
    last_date = pd.to_datetime(df["date"].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)

    # Make results table
    results = pd.DataFrame({"Date": future_dates, "Predicted_" + target: y_future})

    # Optional graph
    if show_graph:
        plt.plot(df["date"], df[target], label="Historical", marker="o")
        plt.plot(results["Date"], results["Predicted_" + target], label="Forecast", marker="x")
        plt.title(f"{days_ahead}-Day Stock Forecast")
        plt.xlabel("Date")
        plt.ylabel(target.capitalize())
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return results