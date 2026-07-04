Smart-Lender/
│
├── app.py
├── train_model.py
├── model.pkl
├── loan_prediction.csv
├── requirements.txt
│
├── templates/
│      ├── index.html
│      └── result.html
│
└── static/
       └── style.css
       pip install flask pandas numpy scikit-learn joblib xgboost matplotlib seaborn
       import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("loan_prediction.csv")

# Fill missing values
df.fillna(df.mode().iloc[0], inplace=True)

# Convert categorical columns
cols = ['Gender','Married','Education','Self_Employed',
        'Property_Area','Loan_Status']

encoder = LabelEncoder()

for col in cols:
    df[col] = encoder.fit_transform(df[col])

df['Dependents'] = df['Dependents'].replace('3+',3)
df['Dependents'] = df['Dependents'].astype(int)

# Features and Target
X = df.drop(['Loan_ID','Loan_Status'],axis=1)
y = df['Loan_Status']

# Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

# Save Model
joblib.dump(model,"model.pkl")

print("Model saved successfully!")
python train_model.py
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

    if prediction[0] == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

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

<input type="number" name="Gender" placeholder="Gender (0=Female,1=Male)" required><br><br>

<input type="number" name="Married" placeholder="Married (0=No,1=Yes)" required><br><br>

<input type="number" name="Dependents" placeholder="Dependents" required><br><br>

<input type="number" name="Education" placeholder="Education (0=Graduate,1=Not Graduate)" required><br><br>

<input type="number" name="Self_Employed" placeholder="Self Employed (0=No,1=Yes)" required><br><br>

<input type="number" name="ApplicantIncome" placeholder="Applicant Income" required><br><br>

<input type="number" name="CoapplicantIncome" placeholder="Coapplicant Income" required><br><br>

<input type="number" name="LoanAmount" placeholder="Loan Amount" required><br><br>

<input type="number" name="Loan_Amount_Term" placeholder="Loan Amount Term" required><br><br>

<input type="number" name="Credit_History" placeholder="Credit History (0 or 1)" required><br><br>

<input type="number" name="Property_Area" placeholder="Property Area (0=Rural,1=Semiurban,2=Urban)" required><br><br>

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

<a href="/">Predict Again</a>

</body>

</html>
cd C:\Users\YourName\Desktop\Smart-Lender
python app.py
* Running on http://127.0.0.1:5000
http://127.0.0.1:5000
| Field              | Example |
| ------------------ | ------- |
| Gender             | 1       |
| Married            | 1       |
| Dependents         | 0       |
| Education          | 0       |
| Self Employed      | 0       |
| Applicant Income   | 5000    |
| Coapplicant Income | 1500    |
| Loan Amount        | 120     |
| Loan Amount Term   | 360     |
| Credit History     | 1       |
| Property Area      | 2       |
Loan Approved
Loan Rejected