Smart_Lender/
│
├── app.py
├── model.pkl
├── train_model.py
├── requirements.txt
├── loan.csv
│
├── templates/
│     ├── index.html
│     └── result.html
│
├── static/
│     ├── style.css
│
├── database/
│     └── smart_lender.db
│
└── sql/
      └── database.sql
      database.sql
      CREATE DATABASE SmartLender;

USE SmartLender;

-- USER TABLE
CREATE TABLE User(
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100),
    email VARCHAR(100),
    role VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- APPLICANT PROFILE
CREATE TABLE Applicant_Profile(
    applicant_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    gender VARCHAR(10),
    married VARCHAR(10),
    education VARCHAR(20),
    self_employed VARCHAR(10),
    dependents INT,
    property_area VARCHAR(30),

    FOREIGN KEY(user_id)
    REFERENCES User(user_id)
);

-- CREDIT HISTORY
CREATE TABLE Credit_History(
    credit_id INT PRIMARY KEY AUTO_INCREMENT,
    applicant_id INT,
    credit_score FLOAT,
    credit_history_status INT,

    FOREIGN KEY(applicant_id)
    REFERENCES Applicant_Profile(applicant_id)
);

-- LOAN APPLICATION
CREATE TABLE Loan_Application(
    loan_id INT PRIMARY KEY AUTO_INCREMENT,
    applicant_id INT,
    income FLOAT,
    coapplicant_income FLOAT,
    loan_amount FLOAT,
    loan_term INT,
    application_date DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(applicant_id)
    REFERENCES Applicant_Profile(applicant_id)
);

-- MODEL TABLE
CREATE TABLE Model(
    model_id INT PRIMARY KEY AUTO_INCREMENT,
    model_name VARCHAR(100),
    model_nm VARCHAR(100),
    algorithm VARCHAR(100),
    training_accuracy FLOAT,
    testing_accuracy FLOAT,
    file_path VARCHAR(200)
);

-- PREDICTION RESULT
CREATE TABLE Prediction_Result(
    prediction_id INT PRIMARY KEY AUTO_INCREMENT,
    loan_id INT,
    model_id INT,
    prediction_status VARCHAR(20),
    probability_score FLOAT,
    prediction_time DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(loan_id)
    REFERENCES Loan_Application(loan_id),

    FOREIGN KEY(model_id)
    REFERENCES Model(model_id)
);
1 : N
1 : 1
1 : N
1 : 1
1 : N
| Field      | Datatype | Description       |
| ---------- | -------- | ----------------- |
| user_id    | INT      | Primary Key       |
| name       | VARCHAR  | User Name         |
| email      | VARCHAR  | Email             |
| role       | VARCHAR  | Applicant/Admin   |
| created_at | DATETIME | Registration Date |
| Field         | Datatype |
| ------------- | -------- |
| applicant_id  | INT      |
| gender        | VARCHAR  |
| married       | VARCHAR  |
| education     | VARCHAR  |
| self_employed | VARCHAR  |
| dependents    | INT      |
| property_area | VARCHAR  |
| Field                 | Datatype |
| --------------------- | -------- |
| credit_id             | INT      |
| credit_score          | FLOAT    |
| credit_history_status | INT      |
| Field              | Datatype |
| ------------------ | -------- |
| loan_id            | INT      |
| income             | FLOAT    |
| coapplicant_income | FLOAT    |
| loan_amount        | FLOAT    |
| loan_term          | INT      |
| Field             | Datatype |
| ----------------- | -------- |
| model_id          | INT      |
| algorithm         | VARCHAR  |
| training_accuracy | FLOAT    |
| testing_accuracy  | FLOAT    |
| Field             | Datatype |
| ----------------- | -------- |
| prediction_id     | INT      |
| prediction_status | VARCHAR  |
| probability_score | FLOAT    |
| prediction_time   | DATETIME |
INSERT INTO User(name,email,role)
VALUES
('Gayathri','gayathri@gmail.com','Applicant');
INSERT INTO Applicant_Profile
(user_id,gender,married,education,self_employed,dependents,property_area)
VALUES
(1,'Female','Yes','Graduate','No',1,'Urban');
INSERT INTO Credit_History
(applicant_id,credit_score,credit_history_status)
VALUES
(1,750,1);
INSERT INTO Loan_Application
(applicant_id,income,coapplicant_income,loan_amount,loan_term)
VALUES
(1,5000,2000,150,360);
INSERT INTO Model
(model_name,model_nm,algorithm,training_accuracy,testing_accuracy,file_path)
VALUES
('Loan Approval','loan_model','XGBoost',98.4,97.9,'model.pkl');
INSERT INTO Prediction_Result
(loan_id,model_id,prediction_status,probability_score)
VALUES
(1,1,'Approved',0.95);
import sqlite3

conn = sqlite3.connect("database/smart_lender.db")
cursor = conn.cursor()

print("Database Connected Successfully")
import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="SmartLender"
)

cursor = conn.cursor()

print("Connected Successfully")