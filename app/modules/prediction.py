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
    """
    Calculate regression coefficients using the Normal Equation method.
    Formula: coefficients = ((x_transpose)*X)^-1 * (x_transpose)*y
    Parameters:
        features (np.array): The input features for training
        target (np.array): The target for prediction

    Returns:
        coefficients (np.array): coefficients [b0, b1, b2, ...]

    """
    try:
        # Add intercept column to the features matrix
        features_with_intercept = add_intercept(features)
        # Apply the Normal Equation: coefficients = ((x_transpose)*X)^-1 * (x_transpose)*y
        x_transpose = features_with_intercept.T
        coefficients = np.linalg.pinv(x_transpose @ features_with_intercept) @ x_transpose @ target
        return coefficients
    except np.linalg.LinAlgError as e:
        print(f"Error in calculate_coefficients(): Error can happen due to perfect correlation in features. {e}")
        raise
    except Exception as e:
        print(f"Error in calculate_coefficients(): Unexpected Error. {e}")
        raise

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

    Parameters:
        data (pd.DataFrame): The historical data to train the model on.
        target_column (str): The name of the column we want to predict.
        n_days (int): The number of future days to predict.

    Outputs:
        Next n_days predictions: Shown in console
        predicted_plot: Plot showing historical and predicted prices
    """
    
    try:
        # Ensure the date column is a datetime object to allow Timedelta addition
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data.dropna(subset=['date'], inplace=True)

        # Ensure n_days is a positive integer
        if not isinstance(n_days, int) or n_days < 1:
            print("Error: Number of days for forecast must be a positive integer.")
            return None
        
        # Ensure the dataframe is not empty
        if data.empty:
            raise ValueError("Error: Input dataframe is empty.")
        
        # Ensure required columns are present
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
        coefficients = calculate_coefficients(features, target) 

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
            # FIX: Reverting the prediction indexing to [0]. Since the model was trained with 1D target 
            # data, predict returns a 1D array of shape (1,). Indexing it once [0] gives the scalar value.
            next_prediction = predict(current_features, coefficients)[0] 
            
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
