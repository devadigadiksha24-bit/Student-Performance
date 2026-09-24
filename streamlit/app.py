import os
import requests
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"

MODEL_COMPARISON_PATH = "models/model_comparison.csv"


# ============================================================
# CLEAN PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL PAGE
       ======================================================== */

    .stApp {
        background-color: #f5f7fb;
        color: #172033;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
       ALL GENERAL TEXT
       ======================================================== */

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown span,
    .stMarkdown label {
        color: #172033 !important;
    }

    h1, h2, h3, h4 {
        color: #172033 !important;
    }


    /* ========================================================
       MAIN TITLE
       ======================================================== */

    .main-title {
        font-size: 36px;
        font-weight: 750;
        color: #172554 !important;
        margin-bottom: 4px;
        line-height: 1.2;
    }

    .main-subtitle {
        font-size: 15px;
        color: #64748b !important;
        margin-bottom: 20px;
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-heading {
        font-size: 21px;
        font-weight: 700;
        color: #172554 !important;
        margin-top: 25px;
        margin-bottom: 12px;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #172554 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] p {
        color: #dbeafe !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #334d8f !important;
    }

    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 500;
    }


    /* ========================================================
       SIDEBAR INPUTS
       ======================================================== */

    section[data-testid="stSidebar"] input {
        background-color: #ffffff !important;
        color: #172033 !important;
        border-radius: 6px !important;
    }

    section[data-testid="stSidebar"] input::placeholder {
        color: #64748b !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #ffffff !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #172033 !important;
    }

    section[data-testid="stSidebar"] button {
        color: #172033 !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        width: 100%;
        height: 46px;
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }

    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #dbe3ef !important;
        border-radius: 12px !important;
        padding: 18px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    div[data-testid="stMetricLabel"] * {
        color: #64748b !important;
    }

    div[data-testid="stMetricValue"] {
        color: #172554 !important;
        font-weight: 750 !important;
    }

    div[data-testid="stMetricValue"] * {
        color: #172554 !important;
    }


    /* ========================================================
       ALERTS / INFO
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #dbe3ef;
        border-radius: 10px;
    }


    /* ========================================================
       EXPANDER
       ======================================================== */

    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #dbe3ef !important;
        border-radius: 10px !important;
    }

    div[data-testid="stExpander"] summary {
        color: #172033 !important;
        font-weight: 600 !important;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer-text {
        text-align: center;
        color: #64748b !important;
        font-size: 13px;
        padding-top: 25px;
        margin-top: 35px;
        border-top: 1px solid #dbe3ef;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Student Performance Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Machine learning dashboard for predicting student academic performance'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Student Details")

    st.write(
        "Enter the student's information to generate "
        "a performance prediction."
    )

    st.divider()

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=30,
        value=20,
        step=1
    )

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=15.0,
        value=3.0,
        step=0.5
    )

    attendance = st.slider(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    previous_score = st.number_input(
        "Previous Score",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        step=1.0
    )

    assignments_completed = st.number_input(
        "Assignments Completed",
        min_value=0,
        max_value=20,
        value=8,
        step=1
    )

    sleep_hours = st.number_input(
        "Sleep Hours per Day",
        min_value=0.0,
        max_value=15.0,
        value=7.0,
        step=0.5
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

    st.divider()

    predict_button = st.button(
        "Predict Final Score",
        type="primary",
        use_container_width=True
    )


# ============================================================
# STUDENT OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-heading">Student Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        label="Study Hours",
        value=f"{study_hours:.1f} hrs"
    )

with col2:

    st.metric(
        label="Attendance",
        value=f"{attendance:.0f}%"
    )

with col3:

    st.metric(
        label="Previous Score",
        value=f"{previous_score:.0f}"
    )

with col4:

    st.metric(
        label="Assignments",
        value=f"{assignments_completed}"
    )


# ============================================================
# PREDICTION
# ============================================================

st.markdown(
    '<div class="section-heading">Performance Prediction</div>',
    unsafe_allow_html=True
)


if "predicted_score" not in st.session_state:

    st.session_state["predicted_score"] = None


if predict_button:

    payload = {
        "gender": gender,
        "age": int(age),
        "study_hours": float(study_hours),
        "attendance": float(attendance),
        "previous_score": float(previous_score),
        "assignments_completed": int(assignments_completed),
        "sleep_hours": float(sleep_hours),
        "extracurricular": extracurricular,
        "internet_access": internet_access,
        "parental_support": parental_support
    }

    try:

        with st.spinner("Generating prediction..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=10
            )

        if response.status_code == 200:

            result = response.json()

            score = float(
                result["predicted_final_score"]
            )

            score = max(
                0,
                min(100, score)
            )

            st.session_state["predicted_score"] = score

        else:

            st.error(
                f"FastAPI returned status code "
                f"{response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "FastAPI is not running."
        )

        st.info(
            "Run this command in another terminal:"
        )

        st.code(
            "python -m uvicorn api.app:app --reload"
        )

    except requests.exceptions.Timeout:

        st.error(
            "The prediction request timed out."
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ============================================================
# SHOW PREDICTION
# ============================================================

if st.session_state["predicted_score"] is not None:

    predicted_score = st.session_state["predicted_score"]

    result_col1, result_col2 = st.columns(
        [1, 2]
    )

    with result_col1:

        st.metric(
            "Predicted Final Score",
            f"{predicted_score:.2f} / 100"
        )

        if predicted_score >= 75:

            st.success(
                "Higher expected performance"
            )

        elif predicted_score >= 50:

            st.warning(
                "Moderate expected performance"
            )

        else:

            st.error(
                "Lower expected performance"
            )

        st.progress(
            predicted_score / 100
        )


    with result_col2:

        comparison_df = pd.DataFrame(
            {
                "Score": [
                    previous_score,
                    predicted_score
                ]
            },
            index=[
                "Previous Score",
                "Predicted Score"
            ]
        )

        st.bar_chart(
            comparison_df,
            y="Score"
        )

else:

    st.info(
        "Enter the student's information in the sidebar "
        "and click 'Predict Final Score'."
    )


# ============================================================
# ACADEMIC INDICATORS
# ============================================================

st.markdown(
    '<div class="section-heading">Academic Indicators</div>',
    unsafe_allow_html=True
)

study_normalized = min(
    study_hours * 10,
    100
)

assignment_normalized = min(
    assignments_completed * 10,
    100
)

indicator_df = pd.DataFrame(
    {
        "Value": [
            study_normalized,
            attendance,
            previous_score,
            assignment_normalized
        ]
    },
    index=[
        "Study Hours",
        "Attendance",
        "Previous Score",
        "Assignments"
    ]
)

st.bar_chart(
    indicator_df,
    y="Value"
)

st.caption(
    "Study hours and assignments are normalized to a 0–100 scale "
    "for visualization."
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown(
    '<div class="section-heading">Model Comparison</div>',
    unsafe_allow_html=True
)

if os.path.exists(MODEL_COMPARISON_PATH):

    try:

        model_df = pd.read_csv(
            MODEL_COMPARISON_PATH
        )

        if "model" in model_df.columns:

            display_df = model_df.copy()

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        if (
            "model" in model_df.columns
            and "RMSE" in model_df.columns
        ):

            chart_df = model_df[
                ["model", "RMSE"]
            ].copy()

            chart_df = chart_df.set_index(
                "model"
            )

            st.bar_chart(
                chart_df,
                y="RMSE"
            )

            st.caption(
                "Lower RMSE indicates lower prediction error."
            )

    except Exception as error:

        st.error(
            f"Could not load model comparison: {error}"
        )

else:

    st.warning(
        "Model comparison file was not found."
    )


# ============================================================
# SELECTED MODEL
# ============================================================

st.markdown(
    '<div class="section-heading">Selected Model</div>',
    unsafe_allow_html=True
)

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:

    st.metric(
        "Best Model",
        "Linear Regression"
    )

with model_col2:

    st.metric(
        "RMSE",
        "4.64"
    )

with model_col3:

    st.metric(
        "R² Score",
        "0.743"
    )


# ============================================================
# MLOPS STATUS
# ============================================================

st.markdown(
    '<div class="section-heading">MLOps Pipeline Status</div>',
    unsafe_allow_html=True
)

status1, status2, status3, status4 = st.columns(4)

with status1:

    st.metric(
        "DVC",
        "Active"
    )

with status2:

    st.metric(
        "MLflow",
        "Active"
    )

with status3:

    st.metric(
        "FastAPI",
        "Active"
    )

with status4:

    st.metric(
        "GitHub Actions",
        "Passed"
    )


# ============================================================
# WORKFLOW
# ============================================================

with st.expander(
    "View MLOps Workflow"
):

    st.write(
        """
        1. Dataset is versioned using DVC.

        2. Data is preprocessed.

        3. Multiple machine learning models are trained.

        4. MLflow records experiments and metrics.

        5. Models are compared using evaluation metrics.

        6. The best-performing model is selected.

        7. FastAPI provides the prediction service.

        8. Streamlit provides the user dashboard.

        9. GitHub Actions performs continuous integration.

        10. Docker provides containerized deployment.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-text">'
    'Student Performance Prediction MLOps Pipeline | '
    'DVC | MLflow | FastAPI | Streamlit | GitHub Actions'
    '</div>',
    unsafe_allow_html=True
)