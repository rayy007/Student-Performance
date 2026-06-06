import streamlit as st
import pandas as pd
import joblib

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("catboost_model.pkl")

model = load_model()

# Outlier treatment for exam_score
def cap_exam_score(value):
    LOWER_BOUND = 20    # Replace with your actual lower bound
    UPPER_BOUND = 100   # Replace with your actual upper bound

    if value < LOWER_BOUND:
        return LOWER_BOUND
    elif value > UPPER_BOUND:
        return UPPER_BOUND
    return value

# Title
st.title("Student Placement Prediction")

st.write("Enter Student Details")

# Inputs
study_hours = st.number_input("Study Hours", min_value=0.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0)
internet_usage = st.number_input("Internet Usage", min_value=0.0)
assignments_completed = st.number_input("Assignments Completed", min_value=0)
previous_score = st.number_input("Previous Score", min_value=0.0)
exam_score = st.number_input("Exam Score", min_value=0.0)

# Prediction
if st.button("Predict"):

    exam_score = cap_exam_score(exam_score)

    input_df = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "sleep_hours": [sleep_hours],
        "internet_usage": [internet_usage],
        "assignments_completed": [assignments_completed],
        "previous_score": [previous_score],
        "exam_score": [exam_score]
    })

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("✅ Prediction: Placed")
    else:
        st.error("❌ Prediction: Not Placed")
