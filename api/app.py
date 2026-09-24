from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


app = FastAPI(
    title="Student Performance Prediction API",
    description="ML API for predicting student final performance",
    version="1.0"
)

model = joblib.load("models/best_model.pkl")


class StudentData(BaseModel):

    gender: str
    age: int
    study_hours: float
    attendance: float
    previous_score: float
    assignments_completed: int
    sleep_hours: float
    extracurricular: str
    internet_access: str
    parental_support: str


@app.get("/")
def home():

    return {
        "message": "Student Performance Prediction API is running"
    }


@app.post("/predict")
def predict(data: StudentData):

    input_data = pd.DataFrame([{
        "age": data.age,
        "study_hours": data.study_hours,
        "attendance": data.attendance,
        "previous_score": data.previous_score,
        "assignments_completed": data.assignments_completed,
        "sleep_hours": data.sleep_hours,

        "gender_Male":
            1 if data.gender == "Male" else 0,

        "extracurricular_Yes":
            1 if data.extracurricular == "Yes" else 0,

        "internet_access_Yes":
            1 if data.internet_access == "Yes" else 0,

        "parental_support_Low":
            1 if data.parental_support == "Low" else 0,

        "parental_support_Medium":
            1 if data.parental_support == "Medium" else 0
    }])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_final_score": round(float(prediction), 2)
    }