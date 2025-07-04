import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv(r'E:\diabetes\diabetes.csv')

# Preprocess
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)

# Streamlit UI
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter the patient's medical information:")

# Input form
with st.form("diabetes_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0.0, step=1.0)
    glucose = st.number_input("Glucose Level", min_value=0.0)
    bp = st.number_input("Blood Pressure", min_value=0.0)
    skin = st.number_input("Skin Thickness", min_value=0.0)
    insulin = st.number_input("Insulin Level", min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0.0, step=1.0)

    submit = st.form_submit_button("Predict")

if submit:
    # Prepare input
    user_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    user_scaled = scaler.transform(user_input)

    # Predict
    prediction = rf_model.predict(user_scaled)

    if prediction[0] == 1:
        st.error("🔴 The model predicts: The person is likely to have diabetes.")
    else:
        st.success("🟢 The model predicts: The person is NOT likely to have diabetes.")
