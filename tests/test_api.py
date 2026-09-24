from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200


def test_prediction():

    data = {
        "gender": "Female",
        "age": 20,
        "study_hours": 5,
        "attendance": 90,
        "previous_score": 75,
        "assignments_completed": 9,
        "sleep_hours": 7,
        "extracurricular": "Yes",
        "internet_access": "Yes",
        "parental_support": "High"
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 200

    result = response.json()

    assert "predicted_final_score" in result