# data_handler manages Data acquisition, Data Loading, Data Cleaning and pre-processing, Data Persistence (store in a CSV)
import pandas as pd

# this handles csv files 
def data_handler(
        filepath: str, 
        filterName: list[str] | None = None,
        filterTime: tuple[int, int] | None = None
    ) -> pd.DataFrame:
    # Parameters:
    #   filepath: path to CSV file
    #   filterName: List of Names to filter, e.g filterName = ['AAPL', 'AMZN', 'GOOG', 'MSFT']
    #   filterTime: Tuple of (start_year, end_year) to filter, e.g filterTime = (2015, 2020)
    df = pd.read_csv(filepath)

    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Remove Missing Values
    df.dropna(inplace=True)

    # Filter specific Names
    if filterName:
        df = df[df['name'].isin(filterName)]

    # Ensure correct data-types, raise errors if encountered
    df['name'] = df['name'].astype(str)
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    for col in 'open','close','high','low','volume':
        df[col] = pd.to_numeric(df[col],errors='raise')

    # Filter by time (years)
    if filterTime:
        start, end = filterTime
        df = df[(df['date'].dt.year >= start) & (df['date'].dt.year <= end)]

    # Ensure 2 decimal points for $
    df = df.round({'open': 2, 'high': 2, 'low': 2, 'close': 2})

    # Sort by name and date, then reset index & drop previous dataframe
    df.sort_values(by=['name', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# this handles pd dataframes from API 
def api_data_handler(
        y_data, 
        filterTime: tuple[int, int] | None = None
    ) -> pd.DataFrame:
    # Parameters:
    #   filepath: path to CSV file
    #   filterName: List of Names to filter, e.g filterName = ['AAPL', 'AMZN', 'GOOG', 'MSFT']
    #   filterTime: Tuple of (start_year, end_year) to filter, e.g filterTime = (2015, 2020)
    df = y_data
    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Remove Missing Values
    df.dropna(inplace=True)

    # Ensure correct data-types, raise errors if encountered
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    for col in 'open','close','high','low','volume':
        df[col] = pd.to_numeric(df[col],errors='raise')

    # Filter by time (years)
    if filterTime:
        start, end = filterTime
        df = df[(df['date'].dt.year >= start) & (df['date'].dt.year <= end)]

    # Ensure 2 decimal points for $
    df = df.round({'open': 2, 'high': 2, 'low': 2, 'close': 2})

    # Sort by name and date, then reset index & drop previous dataframe
    df.sort_values(by=[ 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df