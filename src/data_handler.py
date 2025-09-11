# data_handler manages Data acquisition, Data Loading, Data Cleaning and pre-processing, Data Persistence (store in a CSV)
import pandas as pd
import numpy as np


def data_handler(filepath: str, filterName: list[str]) -> pd.DataFrame:
    # Parameters:
    #   filepath: path to CSV file
    #   filterName: List of Names to filter, e.g filterName = ['AAPL', 'AMZN', 'GOOG', 'MSFT']
    df = pd.read_csv(filepath)

    # Remove Missing Values
    df.dropna(inplace=True)

    # Ensure correct data-types, raise errors if encountered
    df['Name'] = df['Name'].astype(str)
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    df['open'] = pd.to_numeric(df['open'], errors='raise')
    df['close'] = pd.to_numeric(df['close'], errors='raise')
    df['high'] = pd.to_numeric(df['high'], errors='raise')
    df['low'] = pd.to_numeric(df['low'], errors='raise')
    df['volume'] = pd.to_numeric(df['volume'], errors='raise')

    # Ensure 2 decimal points for $
    df = df.round({'open': 2, 'high': 2, 'low': 2, 'close': 2})

    # Filter specific Names
    df = df[df['Name'].isin(filterName)]

    # Sort by name and date, then reset index & drop previous dataframe
    df.sort_values(by=['Name', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df