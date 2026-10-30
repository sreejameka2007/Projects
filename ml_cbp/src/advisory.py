
from pathlib import Path
import sys

# Allow this module to import prediction.py and timetable.py
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prediction import predict_occupancy
from timetable import get_advisory


def analyze_classroom(
    room,
    temperature,
    humidity,
    light,
    co2,
    humidity_ratio,
    timestamp,
):
    """Predict occupancy and generate a timetable-based advisory."""

    prediction = predict_occupancy(
        temperature=temperature,
        humidity=humidity,
        light=light,
        co2=co2,
        humidity_ratio=humidity_ratio,
        timestamp=timestamp,
    )

    advisory = get_advisory(
        room=room,
        timestamp=timestamp,
        predicted_occupied=prediction["Predicted_Occupancy"],
    )

    return {
        **prediction,
        "Room": room,
        "Class_Scheduled": advisory["Class_Scheduled"],
        "Advisory": advisory["Advisory"],
    }


if __name__ == "__main__":
    # Illustrative readings for testing only.
    result = analyze_classroom(
        room="A101",
        temperature=23.0,
        humidity=27.0,
        light=450.0,
        co2=600.0,
        humidity_ratio=0.0045,
        timestamp="2026-10-12 10:30:00",
    )

    print("\nCAMPUS IQ — CLASSROOM ANALYSIS")
    print("=" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 40)
    print("Analysis completed.")
