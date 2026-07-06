import pandas as pd

data = pd.read_csv("loan_prediction.xlsx - loan_prediction.csv")
data.head() 
data.shape
data.columns
data.describe()
data['Loan_Status'].value_counts()