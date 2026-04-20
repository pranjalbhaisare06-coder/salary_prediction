import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Debug: Show files in directory
# -------------------------------
st.write("Available files:", os.listdir())

# -------------------------------
# Load Model Safely
# -------------------------------
MODEL_PATH = "best_model.pkl"

if not os.path.exists('best_model.pkl'):
    st.error(f"Model file '{best_model.pkl}' not found.")
    st.stop()

model = joblib.load(best_model.pkl2)

# -------------------------------
# Load Dataset Safely
# -------------------------------
DATA_PATH = "salary_data.csv"

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset file '{DATA_PATH}' not found.")
    st.stop()

original_df = pd.read_csv(DATA_PATH)

# -------------------------------
# Label Encoding
# -------------------------------
label_encoders = {}

for col in ['Gender', 'Education Level', 'Job Title']:
    le = LabelEncoder()
    le.fit(original_df[col].astype(str))
    label_encoders[col] = le

# -------------------------------
# UI
# -------------------------------
st.title("💰 Salary Prediction App")

age = st.slider("Age", 18, 70, 30)

gender = st.selectbox("Gender", sorted(original_df['Gender'].astype(str).unique()))
education_level = st.selectbox("Education Level", sorted(original_df['Education Level'].astype(str).unique()))
job_title = st.selectbox("Job Title", sorted(original_df['Job Title'].astype(str).unique()))

years_of_experience = st.slider("Years of Experience", 0.0, 40.0, 5.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Salary"):

    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education_level,
        'Job Title': job_title,
        'Years of Experience': years_of_experience
    }])

    # Encode safely
    for col, le in label_encoders.items():
        input_data[col] = input_data[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )

    # Ensure column order
    input_data = input_data[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]

    # Predict
    with st.spinner("Predicting..."):
        prediction = model.predict(input_data)

    st.success(f"💰 Predicted Salary: ₹ {prediction[0]:,.0f}")
