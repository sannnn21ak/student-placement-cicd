
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DATA_PATH = "data/placement_data.csv"
MODEL_PATH = "model.pkl"

FEATURES = [
    "CGPA",
    "Attendance",
    "Coding Score",
    "Projects",
    "Internships",
    "Communication Skills",
]

TARGET = "Placement"


def train_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = set(required_columns) - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if data[required_columns].isnull().any().any():
        raise ValueError("Missing data detected")

    if not data[required_columns].apply(
        lambda column: pd.to_numeric(column, errors="coerce")
    ).notnull().all().all():
        raise ValueError("Invalid non-numeric data detected")

    data = data[required_columns].copy()

    if not data[TARGET].isin([0, 1]).all():
        raise ValueError("Placement must contain only 0 or 1")

    if data[TARGET].nunique() != 2:
        raise ValueError("Dataset must contain both placement classes")

    if not data[FEATURES].apply(
        lambda column: column.between(
            {"CGPA": 0, "Attendance": 0, "Coding Score": 0,
             "Projects": 0, "Internships": 0,
             "Communication Skills": 0}[column.name],
            {"CGPA": 10, "Attendance": 100, "Coding Score": 100,
             "Projects": 100, "Internships": 100,
             "Communication Skills": 100}[column.name],
        )
    ).all().all():
        raise ValueError("Feature values are outside allowed ranges")

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy:.2%}")

    if accuracy < 0.80:
        raise ValueError("Accuracy below 80%. Pipeline stopped.")

    joblib.dump(model, MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not generated")

    print("Model trained successfully")
    print("Model file generated successfully")

    return accuracy


if __name__ == "__main__":
    train_model()
