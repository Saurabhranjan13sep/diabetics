# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:58:38 2024

@author: Saurabh Patil
"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load Dataset
file_path = r'C:\\Users\\Saurabh Patil\\Videos\\dd\\diabetes_prediction_dataset.csv'
data = pd.read_csv(file_path)

# Preprocessing: Handle missing values
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})  # Example encoding
data['smoking_history'] = data['smoking_history'].map({'never': 0, 'current': 1, 'former': 2})  # Example encoding

# Features and target
X = data[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = data['diabetes']

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit App
st.title("Diabetes Prediction System")

st.sidebar.header("Input Features")
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
age = st.sidebar.slider("Age", 0, 100, 25)
hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
smoking_history = st.sidebar.selectbox("Smoking History", ['never', 'current', 'former'])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
hba1c = st.sidebar.slider("HbA1c Level", 4.0, 15.0, 5.0)
blood_glucose = st.sidebar.slider("Blood Glucose Level", 50, 300, 120)

# Map input to numeric
gender = 1 if gender == 'Male' else 0
smoking_history = {'never': 0, 'current': 1, 'former': 2}[smoking_history]

# Predict
user_data = [[gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, blood_glucose]]
prediction = model.predict(user_data)
prediction_text = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

st.write(f"Prediction: {prediction_text}")
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
