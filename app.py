# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import joblib

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

st.title("Heart Disease Risk Prediction")
st.markdown("Using a machine learning model trained on the UCI Heart Disease dataset")

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("final_heart_disease_model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# ------------------- User Inputs -------------------
st.header("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0,1,2,3],
                      format_func=lambda x: ["Typical angina", "Atypical angina",
                                             "Non-anginal pain", "Asymptomatic"][x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                       format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG Results", options=[0,1,2])
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                         format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0,1,2])
    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy (0-4)", 0, 4, 0)
    thal = st.selectbox("Thalassemia", options=[0,1,2,3],
                       format_func=lambda x: {0:"Normal", 1:"Fixed defect",
                                              2:"Reversible defect", 3:"Unknown"}.get(x, "Unknown"))

# ------------------- Prediction -------------------
if st.button("Predict Risk of Heart Disease", type="primary"):
    features = [[age, sex, cp, trestbps, chol, fbs, restecg,
                 thalach, exang, oldpeak, slope, ca, thal]]

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"High Risk – Probability: {probability:.1%}")
        st.warning("Patient likely has heart disease. Recommend clinical review.")
    else:
        st.success(f"Low Risk – Probability: {probability:.1%}")
        st.info("Patient likely does NOT have heart disease.")

# ------------------- Footer -------------------
st.markdown("---")
st.caption("Machine learning model for educational purposes only.")

    except Exception as e:
        st.error(f"Load error: {e}")
        st.stop()
model = load_model()
