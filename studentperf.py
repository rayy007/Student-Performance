# app.py

import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("catboost_model.pkl")

model = load_model()

# --------------------------------------------------
# Outlier Treatment for Exam Score
# Replace these bounds with the values used in training
# --------------------------------------------------
def cap_exam_score(value):
    LOWER_BOUND = 20
    UPPER_BOUND = 100
    return max(LOWER_BOUND, min(value, UPPER_BOUND))

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #1E88E5;
    font-size: 42px;
    font-weight: bold;
}
.sub-title {
    text-align: center;
    color: #666666;
    font-size: 18px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🎓 Student Placement Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Predict whether a student will be placed based on academic performance.</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Input Sliders
# --------------------------------------------------
study_hours = st.slider(
    "📚 Study Hours",
    min_value=0.0,
    max_value=12.0,
    value=5.0,
    step=0.5
)

attendance = st.slider(
    "🏫 Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

sleep_hours = st.slider(
    "😴 Sleep Hours",
    min_value=0.0,
    max_value=12.0,
    value=7.0,
    step=0.5
)

internet_usage = st.slider(
    "🌐 Internet Usage (Hours)",
    min_value=0.0,
    max_value=12.0,
    value=3.0,
    step=0.5
)

assignments_completed = st.slider(
    "📝 Assignments Completed",
    min_value=0,
    max_value=20,
    value=10,
    step=1
)

previous_score = st.slider(
    "📊 Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

exam_score = st.slider(
    "🧾 Exam Score",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🚀 Predict Placement", use_container_width=True):

    # Apply outlier treatment
    exam_score = cap_exam_score(exam_score)

    # Create DataFrame
    input_data = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "sleep_hours": [sleep_hours],
        "internet_usage": [internet_usage],
        "assignments_completed": [assignments_completed],
        "previous_score": [previous_score],
        "exam_score": [exam_score]
    })

    # Prediction
    prediction = model.predict(input_data)
    prediction = prediction[0]

    st.divider()

    # Result
    if prediction == "Placed":
        st.success("🎉 Congratulations! The student is likely to be PLACED.")
        st.balloons()

    elif prediction == "Not Placed":
        st.error("📌 Prediction: The student is likely to be NOT PLACED.")

    else:
        st.info(f"Prediction Result: {prediction}")
