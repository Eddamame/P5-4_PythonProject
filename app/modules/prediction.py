import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from app.modules.visualization import predicted_plot
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
    try:
        # Intercept is needed for b0, to ensure the matrix has the correct dimensions
        # Prediction = b0*intercept + b1*x1 + b2*x2 + ...
        intercept = np.ones((features.shape[0], 1))
        # Add intercept column to the features matrix with np.hstack
        features_with_intercept = np.hstack((intercept, features))
        return features_with_intercept
    except Exception as e:
        print(f"Error in add_intercept(): Error adding intercept column. {e}")
        raise


def calculate_coefficients(features, target):
    # Ensure inputs are NumPy arrays (important defensive programming)
    X = np.array(features)
    Y = np.array(target)
    
    # Check shape: X should be (n_samples, n_features), Y should be (n_samples,)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    model = LinearRegression()
    # Scikit-learn handles all the heavy lifting using optimized NumPy operations
    model.fit(X, Y) 
    
    # Combine coefficient and intercept for simplicity if needed, 
    # but the simplest fix is ensuring you use the Scikit-learn fit/predict methods properly.
    
    # We will return the fitted model object itself to use its .predict() method later
    return model

def predict(features, coefficients):
    """
    Predict values given a feature matrix and fitted coefficients.
    Parameters:
        features (np.array): The input features for prediction.
        coefficients (np.array): The fitted coefficients from the model.

    Returns:
        predictions (np.array): The predicted values

    """
    try:
        # Add intercept column to the features matrix
        features_with_intercept = add_intercept(features)
        # Ensure same number of features and coefficients
        if features_with_intercept.shape[1] != coefficients.shape[0]:
            raise ValueError(f"Error in predict(): Feature and coefficient dimensions do not match. Feature: {features_with_intercept.shape[1]}, Coefficients: {len(coefficients)}")

        # Make predictions: predictions = X * coefficients
        predictions = features_with_intercept @ coefficients
        return predictions
    except ValueError as e:
        print(f"Error in predict(). Different number of features and coefficients. {e}")
    except Exception as e:
        print(f"Error in predict(). Unexpected Error. {e}")

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
    try:
        # Ensure the dataframe is not empty
        if data.empty:
            raise ValueError("Error: Input dataframe is empty.")
        
        # Ensure required columns are present
        required_cols = ['date', target_column]
        if not all(col in data.columns for col in required_cols):
            raise KeyError(f"Error: Dataframe must contain columns: {required_cols}")
        
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
    
    except (ValueError, KeyError) as e:
        print(f"Error in validate_model(): {e}")
    except Exception as e:
        print(f"Error in validate_model(): Unexpected Error. {e}")

def forecast_prices(data, target_column, n_days: int):
    """
    Predicts future prices for a given number of days and plots the result.
    
    Returns:
        str: The HTML div string for the Plotly chart, or None if an error occurs.
    """
    try:
        # Import Plotly utilities inside the function for clean execution
        from plotly.offline import plot
        import numpy as np

        # --- VALIDATION ---
        if not isinstance(n_days, int) or n_days < 1:
            print("Error: Number of days for forecast must be a positive integer.")
            return None
        
        if data.empty:
            raise ValueError("Error: Input dataframe is empty.")
        
        # --- INPUT CONVERSION FIX ---
        # The model requires 'date' and the features: 'open', 'high', 'low', 'volume'
        feature_cols = ['open', 'high', 'low', 'volume']
        required_cols = ['date', target_column] + feature_cols
        
        if not all(col in data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data.columns]
            raise KeyError(f"Error: Dataframe must contain columns. Missing: {missing_cols}. Cannot run predictive model.")
        
        print(f"\n--- Predicting Next {n_days} Day(s) ---")
        
        # Step 1: Prepare data for training
        # Extract features (X) and target (Y) as NumPy arrays
        features = data[feature_cols].values
        target = data[target_column].values
        
        # The 'calculate_coefficients' function must return a model object or coefficients 
        # that work with NumPy arrays. Assuming it returns coefficients/model.
        coefficients = calculate_coefficients(features, target)

        # Step 2: Get the last row of real features to start the prediction loop
        # Ensure the last features are a 2D array for the predict function
        last_known_features = features[-1].reshape(1, -1)
        average_volume = data['volume'].mean()
        last_date = data['date'].iloc[-1]

        # Step 3: Generate future dates for plotting
        # Note: Added import for pd.Timedelta, assuming pandas is imported outside this function
        import pandas as pd
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days).to_pydatetime().tolist()

        # Step 4: Iteratively predict for n_days
        future_predictions = []
        current_features = last_known_features

        for day in range(n_days):
            # next_prediction must be a single float/value
            next_prediction = predict(current_features, coefficients)[0]
            future_predictions.append(next_prediction)
            print(f"Day {day + 1}: Predicted {target_column} = {next_prediction:.2f}")

            # Make previous day's close prediction into next day's features
            # This creates the new features array for the next day's prediction
            next_features = np.array([[
                next_prediction, # Open
                next_prediction, # High
                next_prediction, # Low
                average_volume # Volume (estimated as mean)
            ]])
            current_features = next_features

        print("----------------------------------\n")
        
        # Step 5: Generate and Return Plot HTML
        fig = predicted_plot(data, future_dates, future_predictions)
        
        # Convert the Plotly figure to an HTML div string for Flask rendering
        if fig:
            fig_html_div = plot(fig, output_type='div', include_plotlyjs=False)
            return fig_html_div
        else:
            return None
        
    except Exception as e:
        # The error log showed 'can't multiply sequence by non-int of type 'float'' 
        # which is a math error, likely from mixing NumPy and standard Python lists in `calculate_coefficients` or `predict`.
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