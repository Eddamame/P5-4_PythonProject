import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# Import the LinearRegression class from Scikit-learn
from sklearn.linear_model import LinearRegression 
from plotly.offline import plot # Import plot here for global use
from app.modules.visualization import predicted_plot


"""
The goal of this module is to implement a multiple linear regression model
to predict stock prices based on the features in the dataset.

Features(x) = [Open, High, Low, Volume]
Target(y) = [Close]

Predicted Close Price = b0 + b1*Open + b2*High + b3*Low + b4*Volume

----------------------------------------------------------------------------------------------------------------
VALIDATION - Our goal in validation is to get an evaluation of how accurate our model is
(Steps 1-5 described previously)

----------------------------------------------------------------------------------------------------------------
FORECASTING - Our goal in forecasting is to predict the next day's stock price / future stock prices
(Steps 1-3 described previously)
"""


def calculate_coefficients(features, target):
    """
    Trains and returns a fitted Scikit-learn Linear Regression model.
    This function replaces the need for manual Normal Equation calculation.

    Parameters:
        features (np.array): The input features (X).
        target (np.array): The target values (Y).

    Returns:
        LinearRegression: The fitted Scikit-learn model object.
    """
    try:
        # Ensure inputs are NumPy arrays (important defensive programming)
        X = np.array(features)
        Y = np.array(target)
        
        # Check shape: X should be (n_samples, n_features)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        model = LinearRegression()
        # Scikit-learn handles all the heavy lifting using optimized NumPy operations
        model.fit(X, Y) 
        
        # Return the fitted model object itself
        return model
    except Exception as e:
        print(f"Error in calculate_coefficients: {e}")
        # Re-raising the error here allows the calling function (forecast_prices) to catch it
        raise


def validate_model(data, target_column, test_size=0.2):
    """
    Splits data, trains the model, and evaluates its performance on unseen data.
    
    Returns:
        date_test (pd.Series): test set dates for plotting
        target_test (np.array): Actual target values from test set
        predictions_on_test_data (np.array): Predicted target values for test set
    """
    print("--- Starting Model Validation ---")
    try:
        # Data and column validation remains the same
        if data.empty:
            raise ValueError("Error: Input dataframe is empty.")
        
        required_cols = ['date', target_column, 'open', 'high', 'low', 'volume']
        if not all(col in data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data.columns]
            raise KeyError(f"Error: Dataframe must contain columns. Missing: {missing_cols}")
        
        # Step 1: Prepare features and target
        dates = data['date']
        # IMPORTANT: Use explicit feature selection
        feature_cols = ['open', 'high', 'low', 'volume']
        features = data[feature_cols].values
        target = data[target_column].values

        # Step 2: Split the data
        features_train, features_test, target_train, target_test, date_train, date_test = train_test_split(
            features, target, dates, test_size=test_size, random_state=123
        )
        print(f"Data split into {len(features_train)} training samples and {len(features_test)} testing samples.")

        # Step 3: Train the model
        model = calculate_coefficients(features_train, target_train) # Returns the fitted model object

        # Step 4: Make predictions on the test data using the model's .predict() method
        predictions_on_test_data = model.predict(features_test)

        # Step 5: Evaluate the model
        mse = mean_squared_error(target_test, predictions_on_test_data)
        r2 = r2_score(target_test, predictions_on_test_data)
        
        print("\n--- Model Validation Metrics ---")
        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print(f"R-Squared (R²): {r2:.3f}")
        print("------------------------------\n")

        return date_test, target_test, predictions_on_test_data
    
    except (ValueError, KeyError) as e:
        print(f"Error in validate_model(): {e}")
        return None, None, None
    except Exception as e:
        print(f"Error in validate_model(): Unexpected Error. {e}")
        return None, None, None


def forecast_prices(data, target_column, n_days: int):
    """
    Predicts future prices for a given number of days and plots the result.
    
    Returns:
        str: The HTML div string for the Plotly chart, or None if an error occurs.
    """
    # NOTE: The 'predicted_plot' is already imported globally, so we remove the redundant inner import.
    
    try:
        # Ensure the date column is a datetime object to allow Timedelta addition (from previous fix)
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data.dropna(subset=['date'], inplace=True)

        # --- VALIDATION ---
        if not isinstance(n_days, int) or n_days < 1:
            print("Error: Number of days for forecast must be a positive integer.")
            return None
        
        if data.empty:
            raise ValueError("Error: Input dataframe is empty.")
        
        # --- INPUT PREPARATION ---
        feature_cols = ['open', 'high', 'low', 'volume']
        required_cols = ['date', target_column] + feature_cols
        
        if not all(col in data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data.columns]
            raise KeyError(f"Error: Dataframe must contain columns. Missing: {missing_cols}. Cannot run predictive model.")
        
        print(f"\n--- Predicting Next {n_days} Day(s) ---")
        
        # Step 1: Prepare data and train the model on the entire dataset
        features = data[feature_cols].values
        target = data[target_column].values
        
        # 'model' is the fitted Scikit-learn LinearRegression object
        model = calculate_coefficients(features, target) 

        # Step 2: Get the last row of real features to start the prediction loop
        last_known_features = features[-1].reshape(1, -1) # 2D array (1 sample, 4 features)
        average_volume = data['volume'].mean()
        last_date = data['date'].iloc[-1]

        # Step 3: Generate future dates
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days).to_pydatetime().tolist()

        # Step 4: Iteratively predict for n_days using the model's .predict() method
        future_predictions = []
        current_features = last_known_features

        for day in range(n_days):
            # Use the fitted model object to predict. 
            # FIX: Use [0][0] to correctly extract the scalar value from the 2D array prediction result.
            next_prediction = model.predict(current_features)[0][0] 
            
            future_predictions.append(next_prediction)
            print(f"Day {day + 1}: Predicted {target_column} = {next_prediction:.2f}")

            # Make previous day's close prediction into next day's features
            # This creates the new features array for the next day's prediction
            next_features = np.array([[
                next_prediction, # Open (using last prediction)
                next_prediction, # High (using last prediction)
                next_prediction, # Low (using last prediction)
                average_volume # Volume (estimated as mean)
            ]])
            current_features = next_features

        print("----------------------------------\n")
        
        # Step 5: Generate and Return Plot HTML
        fig = predicted_plot(data, future_dates, future_predictions)
        
        if fig:
            fig_html_div = plot(fig, output_type='div', include_plotlyjs='cdn') # Use 'cdn' for simplicity
            return fig_html_div
        else:
            return None
        
    except Exception as e:
        print(f"An unexpected error occurred during forecasting: {e}")
        return None


    
"""
Notes

add_intercept(features)
    - Calculates intercept column for features matrix
    - Uses features (independent variables)

calculate_coefficients(features, target)
    - Calculates regression coefficients using Normal Equation method
    - Uses features (independent variables) and target (dependent variable)

predict(features, coefficients)
    - Makes predictions using features and coefficients
    - Uses features (independent variables) and coefficients (model parameters)
    - Interdependent functions
        - add_intercept(features)
        - calculate_coefficients(features, target)

validate_model(data, target_column, test_size=0.2)
    - Splits data into train/test, then uses previous functions to calculate predicted values
    - Interdependent functions
        - add_intercept(features)
        - calculate_coefficients(features, target)
        - predict(features, coefficients)

forecast_prices(data, target_column)
    - Predicts future prices for 'n' days
    - Interdependent functions
        - add_intercept(features)
        - calculate_coefficients(features, target)
        - predict(features, coefficients)s
"""
