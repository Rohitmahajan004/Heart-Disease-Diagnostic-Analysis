import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
with open('heart_disease_model.pkl', 'rb') as model_file:
    clf, scaler = pickle.load(model_file)

# Function to predict heart disease
def predict_heart_disease(patient_data):
    patient_data = np.array(patient_data).reshape(1, -1)
    patient_data = scaler.transform(patient_data)
    prediction = clf.predict(patient_data)
    return 'Yes' if prediction[0] == 1 else 'No'

# Streamlit app
st.title("Heart Disease Prediction")

st.write("""
### Enter the patient data:
""")

# Collect user input
age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=50, max_value=250, value=120)
chol = st.number_input("Serum Cholesterol in mg/dl", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: 'True' if x == 1 else 'False')
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: {1: 'Normal', 2: 'Fixed Defect', 3: 'Reversible Defect'}[x])

# Create a list of the input data
patient_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Predict heart disease
if st.button("Predict"):
    result = predict_heart_disease(patient_data)
    st.write(f"The prediction is: **{result}**")

# Run the app using streamlit run app.py
