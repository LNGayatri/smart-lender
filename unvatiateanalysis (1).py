# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
df = pd.read_csv("loan_prediction.csv")
numerical_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]

plt.figure(figsize=(15, 5))

for i, feature in enumerate(numerical_features):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df[feature], kde=True, color="skyblue")
    plt.title(feature)

plt.tight_layout()
plt.show()

categorical_features = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
    "Loan_Status"
]

plt.figure(figsize=(15, 12))

for i, feature in enumerate(categorical_features):
    plt.subplot(3, 3, i + 1)
    sns.countplot(x=feature, data=df)
    plt.title(feature)

plt.tight_layout()
plt.show()
print("\nCategorical Feature Counts:\n")

for feature in categorical_features:
    print(f"\n{feature}")
    print(df[feature].value_counts())