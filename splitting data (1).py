import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    random_state=42
)

df = pd.DataFrame(X, columns=[
    "Income",
    "Age",
    "LoanAmount",
    "CreditScore",
    "EmploymentYears"
])

df["Loan_Status"] = y

print(df.head())
# Features (all columns except Loan_Status)
X = df.drop("Loan_Status", axis=1)

# Target variable
y = df["Loan_Status"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)