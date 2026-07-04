Smart-Lender/

│
├── app.py
├── train_model.py
├── loan_prediction.csv
├── model.pkl
├── requirements.txt
│
├── templates/
│      index.html
│      result.html
│
├── static/
│
└── README.md
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
Loan_Status
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
import joblib
df = pd.read_csv("loan_prediction.csv")

print(df.head())
print(df.info())

print(df.describe())

print(df.isnull().sum())
df.fillna(df.mode().iloc[0], inplace=True)
encoder = LabelEncoder()

columns = ['Gender','Married','Education','Self_Employed',
           'Property_Area','Loan_Status']

for col in columns:
    df[col] = encoder.fit_transform(df[col])

df['Dependents'] = df['Dependents'].replace('3+','3')
df['Dependents'] = df['Dependents'].astype(int)
sns.countplot(x='Loan_Status',data=df)
plt.show()
sns.countplot(x='Gender',data=df)
plt.show()
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
plt.show()
X = df.drop(['Loan_ID','Loan_Status'],axis=1)

y = df['Loan_Status']

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
model = RandomForestClassifier()

model.fit(X_train,y_train)
pred=model.predict(X_test)
accuracy=accuracy_score(y_test,pred)

print("Accuracy:",accuracy)
80%-90%
joblib.dump(model,"model.pkl")
from flask import Flask,render_template,request
import joblib
import numpy as np

app=Flask(__name__)

model=joblib.load("model.pkl")

@app.route('/')

def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])

def predict():

    data=[float(x) for x in request.form.values()]

    prediction=model.predict([data])

    if prediction[0]==1:
        result="Loan Approved"
    else:
        result="Loan Rejected"

    return render_template("result.html",prediction=result)

if __name__=="__main__":
    app.run(debug=True)
    <!DOCTYPE html>
<html>
<head>
<title>Smart Lender</title>
</head>

<body>

<h2>Loan Approval Prediction</h2>

<form action="/predict" method="post">

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

<title>Result</title>

</head>

<body>

<h1>{{prediction}}</h1>

<a href="/">Go Back</a>

</body>

</html>
python train_model.py
python app.py
http://127.0.0.1:5000/
numpy
pandas
matplotlib
seaborn
scikit-learn
flask
joblib