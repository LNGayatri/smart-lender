SmartLender/
│
├── Dataset/
│   ├── loan_prediction.csv
│   ├── loan_prediction.xlsx
│
├── Training/
│   ├── Loan Prediction using ML.ipynb
│
├── Flask/
│   ├── static/
│   │      ├── style.css
│   │
│   ├── templates/
│   │      ├── index.html
│   │      └── result.html
│   │
│   ├── app.py
│   ├── rdf.pkl
│   └── scale1.pkl
│
└── IBM/
app.py
templates/
static/
rdf.pkl
scale1.pkl
                 User

                  │

                  ▼

        HTML / CSS Interface

                  │

                  ▼

            Flask Backend

                  │

        ┌─────────┴──────────┐

        ▼                    ▼

 Load Scaler            Load ML Model

(scale1.pkl)            (rdf.pkl)

        │                    │

        └─────────┬──────────┘

                  ▼

        Loan Prediction

                  │

                  ▼

      Approved / Rejected

                  │

                  ▼

           Display Result
           import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

model = pickle.load(open("rdf.pkl", "rb"))
scaler = pickle.load(open("scale1.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    values = [float(x) for x in request.form.values()]

    final = scaler.transform([values])

    prediction = model.predict(final)

    if prediction[0] == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
    templates/

index.html
result.html
static/

style.css
Loan Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Model Training
      │
      ▼
Save Model (.pkl)
      │
      ▼
Flask Application
      │
      ▼
User Input
      │
      ▼
Prediction
      │
      ▼
Loan Approved / Rejected
