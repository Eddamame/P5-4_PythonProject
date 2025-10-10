import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
"""
The goal of this module is to implement a multiple linear regression model
to predict stock prices based on the features in the dataset.

Features(x) = [Open, High, Low, Volume]
Target(y) = [Close]

Predicted Close Price = b0 + b1*Open + b2*High + b3*Low + b4*Volume

To ensure that the model is accurate, we must split this process into two main parts: Validation and Forecasting

----------------------------------------------------------------------------------------------------------------
VALIDATION - Our goal in validation is to get an evaluation of how accurate our model is

Step 1: Prepare the data
    - Extract features (x) and target (y) from the DataFrame

Step 2: Split the data
    - Split the dataset into training (80%) and testing (20%) sets
    - train_x, train_y for training (80%)
    - test_x, test_y for testing (20%)

Step 3: Train the Model (Find Coefficients)
    - Use train_x and train_y to calculate the coefficients (b0, b1, b2, b3, b4) using Normal Equation method

Step 4: Validate the Model
    - Make predictions based on test_x and test_y
    - Calculate evaluation metrics (MSE, R²)
         - Mean Squared Error (MSE): Measures how wrong the predictions are. e.g Actual Close = 150, Predicted Close = 145, Error = 5, MSE = 25. 
                                     Lower is better (min 0.0)
         - R-Squared (R²): Describes the % of data the model can explain. 0.6 = Explains 60% of the data, 40% unexplained
                           Higher is better (max 1.0)

Step 5: Plot Actual vs. Predicted Values
    - Visualize the performance of the model by plotting actual vs. predicted stock prices

----------------------------------------------------------------------------------------------------------------
FORECASTING - Our goal in forecasting is to predict the next day's stock price / future stock prices

Step 1: Prepare the data
    - Extract features (X) and target (y) from the DataFrame

Step 2: Train the Model (Find Coefficients)
    - Use the entire dataset (X and y) to calculate the coefficients (b0, b1, b2, b3, b4) using Normal Equation method

Step 3: Predict the Next Day's Value
    - Use the most recent data point (last row of X) to predict the next day's stock price

"""

def add_intercept(features):
    """
    Add an intercept column (a column of ones) to the feature matrix.
    This is necessary for calculating the intercept term (b0) in the regression.
    Parameters:
        features (np.array): The input features for training
    Returns:
        features_with_intercept (np.array): The feature matrix with an added intercept column.
    """
    # Intercept is needed for b0, to ensure the matrix has the correct dimensions
    # Prediction = b0*intercept + b1*x1 + b2*x2 + ..., without it, the matrix multiplication won't work
    intercept = np.ones((features.shape[0], 1))
    # Add intercept column to the features matrix with np.hstack
    features_with_intercept = np.hstack((intercept, features))
    return features_with_intercept


def calculate_coefficients(features, target):
    """
    Calculate regression coefficients using the Normal Equation method.
    Formula: coefficients = ((x_transpose)*X)^-1 * (x_transpose)*y
    Parameters:
        features (np.array): The input features for training
        target (np.array): The target for prediction
    Returns:
        coefficients (np.array): coefficients [b0, b1, b2, ...]
    """
    # Add intercept column to the features matrix
    features_with_intercept = add_intercept(features)

    # Apply the Normal Equation: coefficients = ((x_transpose)*X)^-1 * (x_transpose)*y
    x_transpose = features_with_intercept.T
    coefficients = np.linalg.pinv(x_transpose @ features_with_intercept) @ x_transpose @ target
    
    return coefficients


def predict(features, coefficients):
    """
    Predict values given a feature matrix and fitted coefficients.
    Parameters:
        features (np.array): The input features for prediction.
        coefficients (np.array): The fitted coefficients from the model.
    Returns:
        predictions (np.array): The predicted values
    """
    # Add intercept column to the features matrix
    features_with_intercept = add_intercept(features)

    # Make predictions: predictions = X * coefficients
    predictions = features_with_intercept @ coefficients
    return predictions

def validate_model(data, target_column, test_size=0.2):
    """
    Splits data, trains the model, and evaluates its performance on unseen data.
    Parameters:
        data (pd.DataFrame): The input dataframe containing features and target
        target_column (str): The name of the target column in the dataframe
        test_size (float): Proportion of the dataset to include in the test split (default is 0.2)
    Returns:
        date_test (pd.Series): test set dates for plotting
        target_test (np.array): Actual target values from test set
        predictions_on_test_data (np.array): Predicted target values for test set
    """
    print("--- Starting Model Validation ---")
    
    # Step 1: Prepare features and target from the dataframe
    # Features are the inputs (e.g., Open, High, Low, Volume)
    # Target is the output we want to predict (e.g., Close)
    dates = data['date']
    features = data.drop(columns=['date', target_column]).values
    target = data[target_column].values

    # Step 2: Split the data into training and testing sets. Date is also splitted for plotting later
    # random_state=123 to ensure data is split the same way every time
    features_train, features_test, target_train, target_test, date_train, date_test = train_test_split(
        features, target, dates, test_size=test_size, random_state=123
    )
    print(f"Data split into {len(features_train)} training samples and {len(features_test)} testing samples.")

    # Step 3: Train the model by calculating coefficients using only training data(features_train and target_train).
    coefficients = calculate_coefficients(features_train, target_train)

    # Step 4: Make predictions on the test data.
    predictions_on_test_data = predict(features_test, coefficients)

    # Step 5: Evaluate the model by comparing predictions to the actual values.
    mse = mean_squared_error(target_test, predictions_on_test_data)
    r2 = r2_score(target_test, predictions_on_test_data)
    
    print("\n--- Model Validation Metrics ---")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"R-Squared (R²): {r2:.3f}")
    print("------------------------------\n")

    # Return values needed for plotting
    return date_test, target_test, predictions_on_test_data

def forecast_prices(data, target_column):
    """
        Predict future prices for the next 'n' days using an iterative approach.
        This method uses its own predictions to create features for subsequent predictions.

        Parameters:
            data (pd.DataFrame): The historical data to train the model on.
            target_column (str): The name of the column we want to predict.
            n_days (int): The number of future days to predict.

        Returns:
            predictions (list): A list of predicted values for the next 'n' days.
        """
    while True:
        try:
            # Get user input for the number of days
            n_days_input = input("Enter number of days to forecast (or 'q' to quit): ").strip()

            # Allow user to quit
            if n_days_input.lower() in ['q', 'quit']:
                print("Forecast canceled.")
                return []

            # Convert input to an integer
            n_days = int(n_days_input)

            # Validate the input range
            if n_days < 1:
                print("Error: Number of days must be at least 1.")
                continue  # Ask for input again

            if n_days > 30:
                print("Warning: Forecasting more than 30 days ahead is highly unreliable with this model.")
                confirm = input("Do you want to continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue # Ask for input again
            
            # If input is valid, break the loop and proceed to forecasting
            break

        except ValueError:
            print("Invalid input. Please enter a whole number.")
        except KeyboardInterrupt:
            print("\nForecast canceled.")
            return [], []

    # --- Forecasting Logic ---
    try:
        print(f"\n--- Predicting Next {n_days} Day(s) ---")
        
        # Step 1: Train the model on the ENTIRE historical dataset
        features = data.drop(columns=['date', target_column]).values
        target = data[target_column].values
        coefficients = calculate_coefficients(features, target)

        # Step 2: Get the last row of real features to start the prediction loop
        last_known_features = features[-1].reshape(1, -1)
        average_volume = data['volume'].mean()
        last_date = data['date'].iloc[-1]

        # Generate future dates
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days).to_pydatetime().tolist()

        # Step 3: Iteratively predict for n_days
        future_predictions = []
        current_features = last_known_features

        for day in range(n_days):
            next_prediction = predict(current_features, coefficients)[0]
            future_predictions.append(next_prediction)
            print(f"Day {day + 1}: Predicted {target_column} = {next_prediction:.2f}")

            # Create synthetic features for the next iteration
            next_features = np.array([[
                next_prediction,    # Open
                next_prediction,    # High
                next_prediction,    # Low
                average_volume      # Volume
            ]])
            current_features = next_features

        print("----------------------------------\n")
        return future_dates, future_predictions

    except Exception as e:
        print(f"An unexpected error occurred during forecasting: {e}")
        return [], []
    
"""
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