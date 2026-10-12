
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

MODEL_PATH = MODELS_DIR / "occupancy_model.pkl"
MODEL_NAME_PATH = MODELS_DIR / "model_name.txt"

# These column names must match the names used during model training.
FEATURES = [
    "temperature",
    "humidity",
    "light",
    "co2",
    "humidityratio",
    "hour",
    "day_of_week",
]


def load_model():
    """Load the saved occupancy prediction model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_occupancy(
    temperature,
    humidity,
    light,
    co2,
    humidity_ratio,
    timestamp,
):
    """Predict classroom occupancy and return a score if available."""

    timestamp = pd.Timestamp(timestamp)

    values = {
        "temperature": float(temperature),
        "humidity": float(humidity),
        "light": float(light),
        "co2": float(co2),
        "humidityratio": float(humidity_ratio),
        "hour": timestamp.hour,
        "day_of_week": timestamp.dayofweek,
    }

    # Validate input values.
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError(
            "All sensor values must be valid, finite numbers."
        )

    if values["light"] < 0 or values["co2"] <= 0:
        raise ValueError(
            "Light must be non-negative and CO2 must be positive."
        )

    model = load_model()

    # Use the exact lowercase feature names expected by the saved model.
    input_df = pd.DataFrame([values], columns=FEATURES)

    # Predict occupancy.
    prediction = int(model.predict(input_df)[0])

    # Calculate the predicted class probability when supported.
    confidence = None

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if prediction in classes:
                probability = float(
                    probabilities[classes.index(prediction)]
                )

                if np.isfinite(probability):
                    confidence = round(probability * 100, 2)

        except (ValueError, AttributeError, IndexError, TypeError) as exc:
            print(
                "Warning: Model score could not be calculated:",
                str(exc),
            )

    result = {
        "Predicted_Occupancy": prediction,
        "Occupancy_Status": (
            "Occupied" if prediction == 1 else "Not Occupied"
        ),
        "Timestamp": str(timestamp),
        "Model_Score_Percent": confidence,
    }

    return result


if __name__ == "__main__":
    result = predict_occupancy(
        temperature=23.0,
        humidity=27.0,
        light=450.0,
        co2=600.0,
        humidity_ratio=0.0045,
        timestamp="2026-10-12 10:30:00",
    )

    print("\nCAMPUS IQ - Sample Occupancy Prediction")
    print("---------------------------------------")
    print("Timestamp:", result["Timestamp"])
    print("Predicted occupancy:", result["Occupancy_Status"])
    print("Predicted class:", result["Predicted_Occupancy"])

    if result["Model_Score_Percent"] is not None:
        print(
            "Model score:",
            f'{result["Model_Score_Percent"]:.2f}%',
        )
    else:
        print("Model score: Unavailable")

    print("---------------------------------------")
