# app.py

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("catboost_model.pkl")

model = load_model()

def cap_exam_score(value):
    LOWER_BOUND = 20
    UPPER_BOUND = 100
    return max(LOWER_BOUND, min(value, UPPER_BOUND))

st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #4A90E2;
    font-size: 38px;
    font-weight: bold;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: gray;
}
.result-box {
    text-align: center;
    padding: 25px;
    border-radius: 15px;
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎓 Student Placement Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter student details using sliders and predict placement status</div>', unsafe_allow_html=True)

st.divider()

study_hours = st.slider("📚 Study Hours", 0.0, 12.0, 5.0, 0.5)
attendance = st.slider("🏫 Attendance (%)", 0.0, 100.0, 75.0, 1.0)
sleep_hours = st.slider("😴 Sleep Hours", 0.0, 12.0, 7.0, 0.5)
internet_usage = st.slider("🌐 Internet Usage Hours", 0.0, 12.0, 3.0, 0.5)
assignments_completed = st.slider("📝 Assignments Completed", 0, 20, 10, 1)
previous_score = st.slider("📊 Previous Score", 0.0, 100.0, 70.0, 1.0)
exam_score = st.slider("🧾 Exam Score", 0.0, 100.0, 75.0, 1.0)

st.divider()

if st.button("🚀 Predict Placement", use_container_width=True):

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
        st.success("✅ Prediction: Student is Placed")
        st.balloons()
    else:
        st.error("❌ Prediction: Student is Not Placed")
