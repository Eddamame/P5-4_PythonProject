# data_handler manages Data acquisition, Data Loading, Data Cleaning and pre-processing, Data Persistence (store in a CSV)


"""
Fetches raw stock data from the Alpha Vantage API for a list of tickers.

Args:
tickers: A list of stock ticker symbols.
api_key: Your Alpha Vantage API key.

Returns:
A list of dictionaries, where each dictionary is a raw API response.
"""





"""
Cleans and formats the raw API data into a single Pandas DataFrame.

Args:
raw_data_list: A list of raw API response dictionaries.

Returns:
A Pandas DataFrame with cleaned and formatted stock data.
"""



"""
Saves a Pandas DataFrame to a CSV file.

Args:
df: The DataFrame to save.
file_path: The path to save the CSV file to.
"""