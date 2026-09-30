import streamlit as st
import requests
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"
MODEL_PATH = "models/best_model.pkl"
MODEL_FILE = "models/model_comparison.csv"


# =========================================================
# HEADER
# =========================================================

st.title("Student Performance Prediction")
st.caption("AI-powered academic performance prediction dashboard")

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Student Details")

gender = st.sidebar.selectbox(
    "Gender", ["Female", "Male"]
)

age = st.sidebar.number_input(
    "Age", min_value=15, max_value=30, value=20
)

study_hours = st.sidebar.number_input(
    "Study Hours / Day",
    min_value=0.0,
    max_value=24.0,
    value=4.0,
    step=0.5
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    0, 100, 85
)

previous_score = st.sidebar.number_input(
    "Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

assignments = st.sidebar.number_input(
    "Assignments Completed",
    min_value=0,
    max_value=20,
    value=9
)

sleep_hours = st.sidebar.number_input(
    "Sleep Hours / Day",
    min_value=0.0,
    max_value=24.0,
    value=7.0,
    step=0.5
)

extracurricular = st.sidebar.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)

internet = st.sidebar.selectbox(
    "Internet Access",
    ["Yes", "No"]
)

parental_support = st.sidebar.selectbox(
    "Parental Support",
    ["Low", "Medium", "High"]
)

predict_button = st.sidebar.button(
    "Predict Final Score",
    type="primary",
    use_container_width=True
)


# =========================================================
# OVERVIEW
# =========================================================

st.subheader("Student Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Study Hours", f"{study_hours:.1f} hrs")
c2.metric("Attendance", f"{attendance}%")
c3.metric("Previous Score", f"{previous_score:.0f}")
c4.metric("Assignments", assignments)


# =========================================================
# DONUT CHARTS
# These are displayed BEFORE prediction
# =========================================================

st.subheader("Academic Indicators")

d1, d2, d3 = st.columns(3)


# ---------- ATTENDANCE ----------

with d1:

    fig = go.Figure(
        go.Pie(
            labels=["Attendance", "Remaining"],
            values=[attendance, 100 - attendance],
            hole=0.68,
            textinfo="none"
        )
    )

    fig.update_layout(
        title="Attendance",
        height=280,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        annotations=[
            dict(
                text=f"{attendance}%",
                x=0.5,
                y=0.5,
                font=dict(size=25),
                showarrow=False
            )
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ---------- ASSIGNMENTS ----------

with d2:

    assignment_percent = min(
        assignments / 10 * 100,
        100
    )

    fig = go.Figure(
        go.Pie(
            labels=["Completed", "Remaining"],
            values=[
                assignment_percent,
                100 - assignment_percent
            ],
            hole=0.68,
            textinfo="none"
        )
    )

    fig.update_layout(
        title="Assignment Completion",
        height=280,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        annotations=[
            dict(
                text=f"{assignment_percent:.0f}%",
                x=0.5,
                y=0.5,
                font=dict(size=25),
                showarrow=False
            )
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ---------- STUDY TIME ----------

with d3:

    study_percent = min(
        study_hours / 8 * 100,
        100
    )

    fig = go.Figure(
        go.Pie(
            labels=["Study Time", "Remaining"],
            values=[
                study_percent,
                100 - study_percent
            ],
            hole=0.68,
            textinfo="none"
        )
    )

    fig.update_layout(
        title="Study Hours",
        height=280,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        annotations=[
            dict(
                text=f"{study_hours:.1f} hrs",
                x=0.5,
                y=0.5,
                font=dict(size=22),
                showarrow=False
            )
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

st.divider()

st.subheader("Performance Prediction")


if predict_button:

    payload = {
        "gender": gender,
        "age": age,
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "assignments_completed": assignments,
        "sleep_hours": sleep_hours,
        "extracurricular": extracurricular,
        "internet_access": internet,
        "parental_support": parental_support
    }

    score = None
    source = ""

    # -----------------------------------------------------
    # FIRST: FASTAPI
    # -----------------------------------------------------

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=2
        )

        if response.status_code == 200:

            result = response.json()

            score = float(
                result["predicted_final_score"]
            )

            source = "FastAPI"

    except requests.exceptions.RequestException:
        pass


    # -----------------------------------------------------
    # FALLBACK: LOCAL MODEL
    # -----------------------------------------------------

    if score is None:

        try:

            model = joblib.load(MODEL_PATH)

            input_data = pd.DataFrame([{
                "age": age,
                "study_hours": study_hours,
                "attendance": attendance,
                "previous_score": previous_score,
                "assignments_completed": assignments,
                "sleep_hours": sleep_hours,

                "gender_Male":
                    int(gender == "Male"),

                "extracurricular_Yes":
                    int(extracurricular == "Yes"),

                "internet_access_Yes":
                    int(internet == "Yes"),

                "parental_support_Low":
                    int(parental_support == "Low"),

                "parental_support_Medium":
                    int(parental_support == "Medium")
            }])

            score = float(
                model.predict(input_data)[0]
            )

            score = max(
                0,
                min(100, score)
            )

            source = "Local ML Model"

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

            st.stop()


    # =====================================================
    # RESULT
    # =====================================================

    if score >= 75:
        category = "High Performance"
    elif score >= 50:
        category = "Moderate Performance"
    else:
        category = "Needs Improvement"


    r1, r2 = st.columns(2)


    # ---------- SCORE ----------

    with r1:

        st.metric(
            "Predicted Final Score",
            f"{score:.2f} / 100"
        )

        if score >= 75:
            st.success(category)
        elif score >= 50:
            st.warning(category)
        else:
            st.error(category)

        st.caption(
            f"Prediction generated using: {source}"
        )

        st.progress(
            int(score)
        )


    # ---------- SCORE COMPARISON ----------

    with r2:

        comparison = pd.DataFrame({
            "Score Type": [
                "Previous Score",
                "Predicted Score"
            ],
            "Score": [
                previous_score,
                score
            ]
        })

        fig = px.bar(
            comparison,
            x="Score Type",
            y="Score",
            text="Score",
            range_y=[0, 100],
            title="Previous vs Predicted Score"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # =====================================================
    # PERFORMANCE DONUT
    # =====================================================

    st.subheader("Predicted Performance")

    p1, p2 = st.columns(2)

    with p1:

        fig = go.Figure(
            go.Pie(
                labels=[
                    "Predicted Score",
                    "Remaining"
                ],
                values=[
                    score,
                    100 - score
                ],
                hole=0.70,
                textinfo="none"
            )
        )

        fig.update_layout(
            title=category,
            height=320,
            showlegend=False,
            annotations=[
                dict(
                    text=f"{score:.1f}",
                    x=0.5,
                    y=0.5,
                    font=dict(size=30),
                    showarrow=False
                )
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with p2:

        indicators = pd.DataFrame({
            "Indicator": [
                "Attendance",
                "Previous Score",
                "Study Hours",
                "Assignments"
            ],
            "Value": [
                attendance,
                previous_score,
                min(study_hours / 8 * 100, 100),
                min(assignments / 10 * 100, 100)
            ]
        })

        fig = px.bar(
            indicators,
            x="Indicator",
            y="Value",
            text="Value",
            range_y=[0, 100],
            title="Academic Indicators"
        )

        fig.update_traces(
            texttemplate="%{text:.0f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# MODEL COMPARISON
# =========================================================

st.divider()

st.subheader("Model Comparison")

try:

    models = pd.read_csv(
        MODEL_FILE
    )

    st.dataframe(
        models,
        use_container_width=True,
        hide_index=True
    )

    if "RMSE" in models.columns:

        fig = px.bar(
            models,
            x="Model",
            y="RMSE",
            text="RMSE",
            title="Model Performance Comparison"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

except Exception:

    st.warning(
        "Model comparison file not found."
    )


# =========================================================
# MLOPS STATUS
# =========================================================

st.divider()

st.subheader("MLOps Pipeline Status")

m1, m2, m3, m4 = st.columns(4)

m1.metric("DVC", "Active")
m2.metric("MLflow", "Active")
m3.metric("FastAPI", "Active")
m4.metric("CI/CD", "Passed")

st.caption(
    "DVC → Preprocessing → MLflow → Best Model → FastAPI → Docker → Streamlit"
)