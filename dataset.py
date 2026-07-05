| Feature           | Description           |
| ----------------- | --------------------- |
| Loan_ID           | Unique Loan ID        |
| Gender            | Male/Female           |
| Married           | Marital Status        |
| Dependents        | Number of Dependents  |
| Education         | Graduate/Not Graduate |
| Self_Employed     | Employment Status     |
| ApplicantIncome   | Applicant Income      |
| CoapplicantIncome | Co-applicant Income   |
| LoanAmount        | Loan Amount Requested |
| Loan_Amount_Term  | Loan Term (Months)    |
| Credit_History    | Credit History (0/1)  |
| Property_Area     | Urban/Rural/Semiurban |
| Loan_Status       | Target Variable       |
| Property        | Value          |
| --------------- | -------------- |
| Total Records   | 614            |
| Total Features  | 13             |
| Target Variable | Loan_Status    |
| Missing Values  | Yes            |
| Dataset Type    | Structured CSV |
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
import pandas as pd

# Load dataset
df = pd.read_csv("loan.csv")

# Display first five rows
print(df.head())
print(df.info())

print(df.describe())

print(df.shape)

print(df.columns)
print(df.isnull().sum())
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)

df["Married"].fillna(df["Married"].mode()[0], inplace=True)

df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)

df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)

df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)

df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)

df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

columns = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in columns:
    df[col] = encoder.fit_transform(df[col])
    df["Dependents"] = df["Dependents"].replace("3+", 3)
df["Dependents"] = df["Dependents"].astype(int)
df.drop("Loan_ID", axis=1, inplace=True)
X = df.drop("Loan_Status", axis=1)

y = df["Loan_Status"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
from xgboost import XGBClassifier

model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy :", accuracy * 100)
import joblib

joblib.dump(model, "model.pkl")

print("Model Saved Successfully")
Loan Dataset (CSV)
        │
        ▼
Load Dataset (Pandas)
        │
        ▼
Handle Missing Values
        │
        ▼
Encode Categorical Variables
        │
        ▼
Split Train/Test Data
        │
        ▼
Train XGBoost Model
        │
        ▼
Evaluate Accuracy
        │
        ▼
Save model.pkl
        │
        ▼
Flask Web Application
        │
        ▼
Loan Approval Prediction
