# app.py

import streamlit as st
import pandas as pd
import pickle

# Load CatBoost model
with open("catboost_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("CatBoost Classification Prediction App")

st.write("Enter input values below:")

# Example input fields
# Change these according to your dataset columns
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")

if st.button("Predict"):
    input_data = pd.DataFrame({
        "feature1": [feature1],
        "feature2": [feature2],
        "feature3": [feature3],
        "feature4": [feature4]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Class: {prediction[0]}")