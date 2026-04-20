
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('random_forest_regressor_model.pkl')

# Load the original dataset to fit LabelEncoders
try:
    original_df = pd.read_csv('Salary_Data (1).csv')
except FileNotFoundError:
    st.error("Error: 'Salary_Data (1).csv' not found. Please ensure it's in the same directory as the app.")
    st.stop()

# Initialize and fit LabelEncoders for categorical columns
label_encoders = {}
for col in ['Gender', 'Education Level', 'Job Title']:
    le = LabelEncoder()
    # Fit on unique values to handle potential NaN and new values gracefully
    le.fit(original_df[col].astype(str).unique())
    label_encoders[col] = le

# Streamlit App Title
st.title('Salary Prediction App')
st.write('Enter employee details to predict their salary.')

# Input fields for features
age = st.slider('Age', 18, 70, 30)

gender_options = original_df['Gender'].astype(str).unique()
gender = st.selectbox('Gender', sorted(gender_options))

education_options = original_df['Education Level'].astype(str).unique()
education_level = st.selectbox('Education Level', sorted(education_options))

job_title_options = original_df['Job Title'].astype(str).unique()
job_title = st.selectbox('Job Title', sorted(job_title_options))

years_of_experience = st.slider('Years of Experience', 0.0, 40.0, 5.0)

# Predict button
if st.button('Predict Salary'):
    # Create a DataFrame from input values
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education_level,
        'Job Title': job_title,
        'Years of Experience': years_of_experience
    }])

    # Preprocess input data
    for col, le in label_encoders.items():
        # Use .transform, but handle potential unseen labels by converting to str and using .map
        # If a label is unseen, it will result in NaN, which might need further handling depending on model
        # For simplicity, here we'll assume new labels are present in the fitted categories.
        # A more robust solution might involve adding an 'unknown' category during fit.
        input_data[col] = input_data[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
        # If -1 is not a valid encoding, this approach needs refinement.
        # For this example, let's assume all selected values will be known.

    # Ensure all columns are present and in the correct order as during training
    # Assuming the training columns were: 'Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'
    # The LabelEncoder changes the column to numerical, so the `input_data` will have numerical values after transformation.
    # The order of columns matters for prediction, so let's enforce it.
    expected_columns = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
    input_data = input_data[expected_columns]

    # Make prediction
    prediction = model.predict(input_data)

    st.success(f'Predicted Salary: ${prediction[0]:,.2f}')
