# app.py - Heart Disease Prediction App
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from io import BytesIO

# Load model directly from your GitHub repo (raw .pkl URL)
MODEL_URL = "https://raw.githubusercontent.com/dbro-10/heart_disease_streamlit_file/main/final_heart_disease_model.pkl"

@st.cache_resource
def load_model():
    try:
        with st.spinner("Loading model from GitHub..."):
            response = requests.get(MODEL_URL)
            response.raise_for_status()  # Raises error if download fails
            model = joblib.load(BytesIO(response.content))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure the .pkl file is in your repo!")
        st.stop()

model = load_model()
st.success("✅ Model loaded successfully!")

# Page title & info
st.title("🫀 Heart Disease Risk Prediction")
st.markdown("""
**Powered by Random Forest (trained on UCI Heart Disease dataset).**  
Enter patient details to get a personalized risk assessment.
""")

st.sidebar.header("📋 Patient Details")

# User inputs (matching your exact dataset features)
def user_input_features():
    age = st.sidebar.slider("Age (years)", 28, 77, 55)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", 
                              ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 250)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
    restecg = st.sidebar.selectbox("Resting ECG Results", 
                                   ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina?", ["No", "Yes"])
    oldpeak = st.sidebar.slider("ST Depression (exercise)", 0.0, 6.2, 1.0, 0.1)
    slope = st.sidebar.selectbox("ST Segment Slope", ["Upsloping", "Flat", "Downsloping"])
    ca = st.sidebar.slider("Major Vessels (Fluoroscopy, 0-3)", 0, 3, 0)
    thal = st.sidebar.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])

    # Convert to model-ready format (exact column order & encoding from your training)
    data = {
        'age': age,
        'sex': 1 if sex == "Male" else 0,
        'cp': ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp),
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == "Yes" else 0,
        'restecg': ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg),
        'thalach': thalach,
        'exang': 1 if exang == "Yes" else 0,
        'oldpeak': oldpeak,
        'slope': ["Upsloping", "Flat", "Downsloping"].index(slope),
        'ca': ca,
        'thal': ["Normal", "Fixed defect", "Reversible defect"].index(thal)
    }
    return pd.DataFrame([data])  # Note: No 'id' column, as dropped in training

input_df = user_input_features()

# Show inputs
with st.expander("👁️ Review Your Inputs"):
    st.write(input_df.T)

# Predict button
if st.button("🔮 Predict Risk", type="primary"):
    with st.spinner("Analyzing heart health..."):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100

    st.markdown("### 📊 **Results**")
    if prediction == 1:
        st.error(f"🚨 **HIGH RISK** of Heart Disease")
        st.warning(f"Estimated Probability: **{probability:.1f}%**")
        st.info("💡 Recommendation: Consult a cardiologist immediately for further tests like ECG or stress test.")
    else:
        st.success(f"✅ **LOW RISK** of Heart Disease")
        st.info(f"Estimated Probability: **{probability:.1f}%**")
        st.info("💡 Tip: Maintain healthy lifestyle — exercise, balanced diet, regular check-ups.")

    # Bonus: Top features importance
    st.markdown("### 📈 Key Risk Factors (Model Insights)")
    importances = pd.Series(model.feature_importances_, index=input_df.columns).sort_values(ascending=False)
    st.bar_chart(importances.head(6))

# Footer
st.markdown("---")
st.caption("*Built with Streamlit | Model Accuracy: ~85% on test data | Not medical advice — for educational use only.*")
