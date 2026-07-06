Feature_Scaling_Project
feature_scaling.py
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Creating a sample dataset

X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    random_state=42
)

# Convert to DataFrame

df = pd.DataFrame(X, columns=[
    "Age",
    "Salary",
    "Experience",
    "Loan",
    "CreditScore"
])

df["Target"] = y

print("First Five Rows")
print(df.head())
# Features

X = df.drop("Target", axis=1)

# Target

y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
# Create StandardScaler object

scaler = StandardScaler()

# Fit and transform training data

X_train_scaled = scaler.fit_transform(X_train)

# Transform testing data

X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)

print("\nScaled Training Data")

print(X_train_scaled.head())
print("\nOriginal Data")

print(X.head())

print("\nScaled Data")

print(X_train_scaled.head())
plt.boxplot(X)

plt.title("Before Feature Scaling")

plt.xticks(range(1,6), X.columns)

plt.show()
plt.figure(figsize=(8,5))

plt.boxplot(X_train_scaled)

plt.title("After Feature Scaling")

plt.xticks(range(1,6), X.columns)
print("\nTraining Shape")

print(X_train_scaled.shape)

print("\nTesting Shape")

print(X_test_scaled.shape)