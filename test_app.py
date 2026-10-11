
from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={
            "cgpa": 8.5,
            "attendance": 92,
            "coding_score": 85,
            "projects": 3,
            "internships": 2,
            "communication_skills": 88,
        },
    )

    assert response.status_code == 200
    assert response.json()["prediction"] in [
        "Placed",
        "Not Placed",
    ]


def test_invalid_input():
    response = client.post(
        "/predict",
        json={
            "cgpa": 12,
            "attendance": 92,
            "coding_score": 85,
            "projects": 3,
            "internships": 2,
            "communication_skills": 88,
        },
    )

    assert response.status_code == 422
