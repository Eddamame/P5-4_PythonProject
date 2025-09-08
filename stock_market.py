#Import Libraries
import pandas as pd
import numpy as np

#Read CSV File from github
dataset_url = 'https://github.com/Eddamame/INF1002_Stock/blob/main/Dataset/StockAnalysisDataset.csv?raw=true'
df = pd.read_csv(dataset_url)

#Change all floats to 2 decimal places
df = df.round({'open': 2, 'high': 2, 'low': 2, 'close': 2})

#Convert 'date' column into datetime format (YYYY-MM-DD)
df['date'] = pd.to_datetime(df['date'])

<<<<<<< HEAD
# Notes for reading from pandas (pd)
# print(df.info()) -- Checks datatype of each column
# print(df.head(5)) -- Shows first 5 rows
# print(df.Name.value_counts()) -- Tells you how many rows per 'Name'
# REMOVE NOTE FOR FINAL SUBMISSION ^

# For those who need to use graphs,
# Consider using libraries such as 'matplotlib','seaborn', 'plotnine' -- u guys can choose
=======
#Ignatius Update
#Lideon Update
>>>>>>> refs/remotes/origin/main
