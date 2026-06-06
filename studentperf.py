import streamlit as st
import pandas as pd
import joblib
import base64

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

# --------------------------------------------------
# Background Image
# --------------------------------------------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    bg_image = get_base64("background.jpg")

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}

        .title {{
            text-align:center;
            color:white;
            font-size:42px;
            font-weight:bold;
        }}

        .subtitle {{
            text-align:center;
            color:white;
            font-size:18px;
        }}

        .block-container {{
            background-color: rgba(255,255,255,0.88);
            padding: 2rem;
            border-radius: 15px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

except:
    pass

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("catboost_model.pkl")

model = load_model()

# --------------------------------------------------
# Outlier Handling
# --------------------------------------------------
def cap_exam_score(value):
    LOWER_BOUND = 20
    UPPER_BOUND = 100
    return max(LOWER_BOUND, min(value, UPPER_BOUND))

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<div class="title">🎓 Student Placement Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict student placement using Machine Learning</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# --------------------------------------------------
# Inputs
# --------------------------------------------------
study_hours = st.slider(
    "📚 Study Hours",
    0.0, 12.0, 5.0, 0.5
)

attendance = st.slider(
    "🏫 Attendance (%)",
    0.0, 100.0, 75.0, 1.0
)

sleep_hours = st.slider(
    "😴 Sleep Hours",
    0.0, 12.0, 7.0, 0.5
)

internet_usage = st.slider(
    "🌐 Internet Usage (Hours)",
    0.0, 12.0, 3.0, 0.5
)

assignments_completed = st.slider(
    "📝 Assignments Completed",
    0, 20, 10
)

previous_score = st.slider(
    "📊 Previous Score",
    0.0, 100.0, 70.0, 1.0
)

exam_score = st.slider(
    "🧾 Exam Score",
    0.0, 100.0, 75.0, 1.0
)

st.markdown("---")

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
if st.button("🚀 Predict Placement", use_container_width=True):

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
    prediction = prediction[0]

    if str(prediction).strip().lower() == "placed":
        st.success("🎉 Prediction: Student is likely to be PLACED")
        st.balloons()

    elif str(prediction).strip().lower() == "not placed":
        st.error("❌ Prediction: Student is likely to be NOT PLACED")

    else:
        st.info(f"Prediction: {prediction}")
