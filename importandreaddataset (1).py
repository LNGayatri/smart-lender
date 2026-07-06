import pandas as pd
data = pd.read_csv("loan_prediction.csv")
print(data.head())
print(data.shape)
data.info()
print(data.describe())
print(data.columns)
print(data.isnull().sum())