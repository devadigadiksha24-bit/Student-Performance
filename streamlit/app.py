import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# Load model
# -----------------------------

model = joblib.load("models/best_model.pkl")


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #666;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}

.result {
    background-color: #eef6ff;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #cfe3ff;
}

.score {
    font-size: 48px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="title">Student Performance Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict student final academic performance using a machine learning model.'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Input section
# -----------------------------

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("Student Information")

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    age = st.number_input(
        "Age",
        min_value=17,
        max_value=25,
        value=20
    )

    study_hours = st.slider(
        "Daily Study Hours",
        1.0,
        10.0,
        5.0,
        0.5
    )

with col2:

    attendance = st.slider(
        "Attendance (%)",
        40.0,
        100.0,
        80.0,
        1.0
    )

    previous_score = st.slider(
        "Previous Score",
        30.0,
        100.0,
        70.0,
        1.0
    )

    assignments_completed = st.slider(
        "Assignments Completed",
        0,
        10,
        7
    )

with col3:

    sleep_hours = st.slider(
        "Sleep Hours",
        4.0,
        10.0,
        7.0,
        0.5
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )

    internet_access = st.selectbox(
        "Internet Access",
        ["Yes", "No"]
    )

parental_support = st.selectbox(
    "Parental Support",
    ["Low", "Medium", "High"]
)

st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# Prediction
# -----------------------------

if st.button(
    "Predict Final Score",
    use_container_width=True
):

    input_data = pd.DataFrame([{

        "age": age,
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "assignments_completed": assignments_completed,
        "sleep_hours": sleep_hours,

        "gender_Male":
            1 if gender == "Male" else 0,

        "extracurricular_Yes":
            1 if extracurricular == "Yes" else 0,

        "internet_access_Yes":
            1 if internet_access == "Yes" else 0,

        "parental_support_Low":
            1 if parental_support == "Low" else 0,

        "parental_support_Medium":
            1 if parental_support == "Medium" else 0
    }])

    prediction = model.predict(input_data)[0]

    prediction = max(
        0,
        min(100, prediction)
    )

    st.markdown(
        '<div class="result">',
        unsafe_allow_html=True
    )

    st.subheader("Predicted Final Score")

    st.markdown(
        f'<div class="score">{prediction:.2f}%</div>',
        unsafe_allow_html=True
    )

    if prediction >= 75:

        st.success(
            "The predicted performance is in the higher score range."
        )

    elif prediction >= 50:

        st.info(
            "The predicted performance is in the moderate score range."
        )

    else:

        st.warning(
            "The predicted performance is in the lower score range."
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# -----------------------------
# Project information
# -----------------------------

st.divider()

st.subheader("MLOps Pipeline")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Data Versioning", "DVC")
col2.metric("Experiment Tracking", "MLflow")
col3.metric("Model Serving", "FastAPI")
col4.metric("Dashboard", "Streamlit")

st.caption(
    "Student Performance Prediction | MLOps Academic Project"
)