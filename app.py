
import os

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = "model.pkl"

FEATURES = [
    "CGPA",
    "Attendance",
    "Coding Score",
    "Projects",
    "Internships",
    "Communication Skills",
]

app = FastAPI(title="Student Placement Prediction API")


class Student(BaseModel):
    cgpa: float = Field(ge=0, le=10)
    attendance: float = Field(ge=0, le=100)
    coding_score: float = Field(ge=0, le=100)
    projects: int = Field(ge=0, le=100)
    internships: int = Field(ge=0, le=100)
    communication_skills: float = Field(ge=0, le=100)


@app.get("/")
def home():
    return {"message": "Student Placement Prediction API is running"}


@app.post("/predict")
def predict(student: Student):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model is not available")

    model = joblib.load(MODEL_PATH)

    values = [[
        student.cgpa,
        student.attendance,
        student.coding_score,
        student.projects,
        student.internships,
        student.communication_skills,
    ]]

    prediction = int(model.predict(values)[0])

    return {
        "prediction": "Placed" if prediction == 1 else "Not Placed"
    }
