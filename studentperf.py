# app.py

import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# -----------------------------
# Load CatBoost Model
# -----------------------------
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    return model

model = load_model()

# -----------------------------
# Outlier Capping Function
# -----------------------------
def cap_exam_score(value):
    EXAM_SCORE_LOWER = 20   # replace with your training lower bound
    EXAM_SCORE_UPPER = 100  # replace with your training upper bound

    if value < EXAM_SCORE_LOWER:
        return EXAM_SCORE_LOWER
    elif value > EXAM_SCORE_UPPER:
        return EXAM_SCORE_UPPER
    else:
        return value

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Student Placement Prediction")
st.write("Enter student details below:")

study_hours = st.number_input("Study Hours", min_value=0.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0)
internet_usage = st.number_input("Internet Usage Hours", min_value=0.0)
assignments_completed = st.number_input("Assignments Completed", min_value=0)
previous_score = st.number_input("Previous Score", min_value=0.0)
exam_score = st.number_input("Exam Score", min_value=0.0)

if st.button("Predict"):

    exam_score = cap_exam_score(exam_score)

    input_data = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "sleep_hours": [sleep_hours],
        "internet_usage": [internet_usage],
        "assignments_completed": [assignments_completed],
        "previous_score": [previous_score],
        "exam_score": [exam_score]
    })

    prediction = model.predict(input_data)

    if int(prediction[0]) == 1:
        st.success("Prediction: Placed")
    else:
        st.error("Prediction: Not Placed")