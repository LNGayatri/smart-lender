import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("loan_prediction.csv")
print("Dataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())
print("\nMissing Values Before Treatment:")
print(df.isnull().sum())
numerical_columns = ['LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome', 'CoapplicantIncome']

for col in numerical_columns:
    if col in df.columns:
        df[col].fillna(df[col].mean(), inplace=True)
categorical_columns = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'Credit_History',
    'Property_Area',
    'Loan_Status'
]

for col in categorical_columns:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
label_encoder = LabelEncoder()

for col in categorical_columns:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])
print("\nMissing Values After Treatment:")
print(df.isnull().sum())
print("\nEncoded Dataset:")
print(df.head())
print("\nUpdated Dataset Information:")
print(df.info())