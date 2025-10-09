import numpy as np
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
    Fit regression coefficients using the Normal Equation method.
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
    """
    print("--- Starting Model Validation ---")
    
    # Step 1: Prepare features and target from the dataframe
    # Features are the inputs (e.g., Open, High, Low, Volume)
    # Target is the output we want to predict (e.g., Close)
    features = data.drop(columns=['date', 'name', target_column]).values
    target = data[target_column].values

    # Step 2: Split the data into training and testing sets.
    # random_state=123 to ensure data is split the same way every time
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=test_size, random_state=123
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

    # Return the actual and predicted values for plotting if needed
    return target_test, predictions_on_test_data


def forecast_prices(data, target_column, n_days=1):
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
    print(f"--- Predicting Next {n_days} Day(s) ---")
    
    # Step 1: Train the model on the ENTIRE historical dataset to get the best coefficients.
    features = data.drop(columns=['date', 'name', target_column]).values
    target = data[target_column].values
    coefficients = calculate_coefficients(features, target)

    # Step 2: Get the last row of real features to start the prediction loop.
    last_known_features = features[-1].reshape(1, -1)

    # We need to make an assumption for future volume. A simple one is to use the average historical volume.
    average_volume = data['volume'].mean()

    # Step 3: Iteratively predict for n_days.
    future_predictions = []
    current_features = last_known_features

    for day in range(n_days):
        # Predict the next day's value
        next_prediction = predict(current_features, coefficients)[0]
        future_predictions.append(next_prediction)
        print(f"Day {day + 1}: Predicted {target_column} = {next_prediction:.2f}")

        # Create synthetic features for the *next* prediction in the loop.
        # Assumption: The next day's Open, High, and Low will be the predicted Close price.
        # This is a major simplification and the main reason why this forecasting method is not accurate for many days ahead.
        next_features = np.array([[
            next_prediction,    # Open
            next_prediction,    # High
            next_prediction,    # Low
            average_volume      # Volume
        ]])
        
        # Update the features for the next iteration of the loop.
        current_features = next_features

    print("----------------------------------\n")
    return future_predictions