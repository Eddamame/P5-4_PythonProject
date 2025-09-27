import numpy as np

# Predictive Modelling

def slr(x: list[float], y: list[float]) -> tuple[float, float]:
    # Parameters:
    #   x = independent variable 
    #   y = dependent variable (the one you want to predict)
    n = len(x)
    if n!= len(y):
        raise ValueError("x and y must have the same number of rows")
    if n == 0:
        raise ValueError("No values found")
    
    #Calculate mean for both x and y
    x_mean = sum(x)/n
    y_mean = sum(y)/n

    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    if denominator == 0:
        raise ValueError("Cannot compute slope (all x values identical)")
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    return slope, intercept

def predict(x_new: list[float], slope: float, intercept: float) -> list[float]:
    return [slope * xi + intercept for xi in x_new]
