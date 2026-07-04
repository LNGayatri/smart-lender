Smart-Lender/
│
├── app.py
├── train_model.py
├── model.pkl
├── loan_prediction.csv
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   └── style.css
│
└── screenshots
pip install numpy pandas matplotlib seaborn scikit-learn flask joblib xgboost
Loan_ID
Gender
Married
Dependents
Education
Self_Employed
ApplicantIncome
CoapplicantIncome
LoanAmount
Loan_Amount_Term
Credit_History
Property_Area
Loan_Status
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv("loan_prediction.csv")

print(df.head())

print(df.info())

print(df.describe())
print(df.isnull().sum())
df.fillna(df.mode().iloc[0], inplace=True)
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
sns.countplot(x='Loan_Status', data=df)
plt.show()
sns.histplot(df['ApplicantIncome'])
plt.show()
sns.countplot(x='Education', hue='Loan_Status', data=df)
plt.show()
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()
encoder = LabelEncoder()

cols = [
    'Gender',
    'Married',
    'Education',
    'Self_Employed',
    'Property_Area',
    'Loan_Status'
]

for col in cols:
    df[col] = encoder.fit_transform(df[col])

df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = df['Dependents'].astype(int)
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)

y = df['Loan_Status']

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Decision Tree:", accuracy_score(y_test, dt_pred))
rf = RandomForestClassifier()

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest:", accuracy_score(y_test, rf_pred))
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

print("KNN:", accuracy_score(y_test, knn_pred))
xgb = XGBClassifier()

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

print("XGBoost:", accuracy_score(y_test, xgb_pred))
joblib.dump(rf, "model.pkl")
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(x) for x in request.form.values()]
    prediction = model.predict([values])

    result = "Loan Approved" if prediction[0] == 1 else "Loan Rejected"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
    <!DOCTYPE html>
<html>
<head>
    <title>Smart Lender</title>
</head>
<body>

<h2>Loan Approval Prediction</h2>

<form action="/predict" method="POST">

<input type="text" name="Gender" placeholder="Gender"><br><br>

<input type="text" name="Married" placeholder="Married"><br><br>

<input type="text" name="Dependents" placeholder="Dependents"><br><br>

<input type="text" name="Education" placeholder="Education"><br><br>

<input type="text" name="Self_Employed" placeholder="Self Employed"><br><br>

<input type="text" name="ApplicantIncome" placeholder="Applicant Income"><br><br>

<input type="text" name="CoapplicantIncome" placeholder="Coapplicant Income"><br><br>

<input type="text" name="LoanAmount" placeholder="Loan Amount"><br><br>

<input type="text" name="Loan_Amount_Term" placeholder="Loan Amount Term"><br><br>

<input type="text" name="Credit_History" placeholder="Credit History"><br><br>

<input type="text" name="Property_Area" placeholder="Property Area"><br><br>

<button type="submit">Predict</button>

</form>

</body>
</html>
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body>

<h2>{{ prediction }}</h2>

<a href="/">Predict Another Loan</a>

</body>
</html>
python train_model.py
python app.py
http://127.0.0.1:5000/