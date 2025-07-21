import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict whether they will churn.")

# User inputs
def get_input():
    CreditScore = st.slider('Credit Score', 300, 900, 600)
    Geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Age = st.slider('Age', 18, 100, 35)
    Tenure = st.slider('Tenure (Years)', 0, 10, 5)
    Balance = st.number_input('Balance', min_value=0.0, max_value=300000.0, value=50000.0)
    NumOfProducts = st.selectbox('Number of Products', [1, 2, 3, 4])
    HasCrCard = st.selectbox('Has Credit Card?', [1, 0])
    IsActiveMember = st.selectbox('Is Active Member?', [1, 0])
    EstimatedSalary = st.number_input('Estimated Salary', min_value=0.0, max_value=200000.0, value=50000.0)

    # Manual encoding
    geo_map = {'France': 0, 'Germany': 1, 'Spain': 2}
    gender_map = {'Female': 0, 'Male': 1}

    input_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Geography': [geo_map[Geography]],
        'Gender': [gender_map[Gender]],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary]
    })

    return input_data

# Get input from user
input_df = get_input()

# Scale input
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_input)
    prob = model.predict_proba(scaled_input)[0][1]

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"Customer will churn. (Probability: {prob:.2%})")
    else:
        st.success(f"Customer will stay. (Probability: {prob:.2%})")
