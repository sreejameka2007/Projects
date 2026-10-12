
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TIMETABLE_PATH = ROOT / "data" / "timetable.csv"


def load_timetable():
    """Load and validate the classroom timetable."""
    if not TIMETABLE_PATH.exists():
        raise FileNotFoundError(
            f"Timetable not found: {TIMETABLE_PATH}"
        )

    timetable = pd.read_csv(TIMETABLE_PATH)
    timetable.columns = timetable.columns.str.strip()

    required = {
        "Day", "Room", "Start_Time",
        "End_Time", "Class_Scheduled"
    }

    missing = required - set(timetable.columns)
    if missing:
        raise ValueError(
            f"Timetable is missing columns: {sorted(missing)}"
        )

    timetable["Day"] = timetable["Day"].astype(str).str.strip()
    timetable["Room"] = timetable["Room"].astype(str).str.strip()
    timetable["Start_Time"] = timetable["Start_Time"].astype(str).str.strip()
    timetable["End_Time"] = timetable["End_Time"].astype(str).str.strip()

    timetable["Class_Scheduled"] = (
        timetable["Class_Scheduled"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": True,
            "true": True,
            "1": True,
            "no": False,
            "false": False,
            "0": False,
        })
    )

    if timetable["Class_Scheduled"].isna().any():
        raise ValueError(
            "Class_Scheduled must contain Yes/No or True/False values."
        )

    for column in ["Start_Time", "End_Time"]:
        parsed = pd.to_datetime(
            timetable[column],
            format="%H:%M",
            errors="coerce"
        )
        if parsed.isna().any():
            raise ValueError(
                f"{column} must use 24-hour HH:MM format."
            )
        timetable[column] = parsed.dt.time

    if (
        timetable["Start_Time"] >= timetable["End_Time"]
    ).any():
        raise ValueError(
            "Each timetable entry must end after it starts."
        )

    return timetable


def get_advisory(room, timestamp, predicted_occupied):
    """
    Compare predicted occupancy with the scheduled class.

    timestamp: datetime or a string such as '2026-10-12 10:30'
    predicted_occupied: Boolean or 0/1
    """
    timetable = load_timetable()

    timestamp = pd.Timestamp(timestamp)
    day_name = timestamp.day_name()
    current_time = timestamp.time()

    room_entries = timetable[
        (timetable["Room"].str.casefold() == str(room).strip().casefold())
        & (timetable["Day"].str.casefold() == day_name.casefold())
        & (timetable["Start_Time"] <= current_time)
        & (timetable["End_Time"] > current_time)
        & (timetable["Class_Scheduled"])
    ]

    scheduled = not room_entries.empty
    occupied = bool(predicted_occupied)

    if scheduled and occupied:
        advisory = "Normal"
    elif not scheduled and occupied:
        advisory = "Unexpected Occupancy"
    elif scheduled and not occupied:
        advisory = "Possible Energy-Saving Check"
    else:
        advisory = "Normal — No Class Scheduled"

    return {
        "Room": room,
        "Timestamp": str(timestamp),
        "Class_Scheduled": scheduled,
        "Predicted_Occupied": occupied,
        "Advisory": advisory,
    }


if __name__ == "__main__":
    timetable = load_timetable()
    print("Timetable loaded successfully.")
    print(f"Number of timetable entries: {len(timetable)}")
    print("\nFirst five entries:")
    print(timetable.head().to_string(index=False))
