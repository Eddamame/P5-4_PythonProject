#Import Libraries
import pandas as pd
import numpy as np

#Read CSV File from github
dataset_url = 'https://github.com/Eddamame/INF1002_Stock/blob/main/Dataset/StockAnalysisDataset.csv?raw=true'
df = pd.read_csv(dataset_url)

#Exploratory Data Analysis (EDA) - Understanding your Data

#First Look at Dataset
print(df.head(5))

#How many entries per Stock? 'Name' = column with Stock Name
print(df.Name.value_counts())

#Check Data Types of Columns
print(df.info())

#Change all floats to 2 decimal places
df = df.round({'open': 2, 'high': 2, 'low': 2, 'close': 2})

#Check updated dataset
print(df.head(7))

#Test Main
