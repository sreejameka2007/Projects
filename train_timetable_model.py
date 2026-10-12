
from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

BASE = Path(__file__).parent
DATA = BASE / "data"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

TRAIN_FILE = DATA / "datatraining (1).txt"
TEST_FILES = [
    DATA / "datatest (1).txt",
    DATA / "datatest2 (1).txt",
]
TIMETABLE_FILE = DATA / "timetable.csv"

# Read the sample timetable.
schedule = pd.read_csv(TIMETABLE_FILE)
schedule.columns = schedule.columns.str.strip()

schedule["Day"] = schedule["Day"].str.strip()
schedule["Start_Time"] = schedule["Start_Time"].astype(str)
schedule["End_Time"] = schedule["End_Time"].astype(str)

# Use A101 as the illustrative monitored room.
schedule = schedule[schedule["Room"] == "A101"].copy()

schedule["Class_Scheduled"] = pd.to_numeric(
    schedule["Class_Scheduled"], errors="raise"
).astype(int)

def add_timetable_feature(df):
    df = df.copy()

    df.columns = [
        str(c).strip().strip('"').lower()
        for c in df.columns
    ]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "temperature", "humidity", "light",
        "co2", "humidityratio", "occupancy"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"] + numeric_cols).copy()

    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_name"] = df["date"].dt.day_name()
    df["time_of_day"] = df["date"].dt.strftime("%H:%M")

    # Default to no class when no matching A101 slot exists.
    df["class_scheduled"] = 0

    for _, slot in schedule.iterrows():
        start = slot["Start_Time"]
        end = slot["End_Time"]

        mask = (
            (df["day_name"] == slot["Day"])
            & (df["time_of_day"] >= start)
            & (df["time_of_day"] < end)
        )

        df.loc[mask, "class_scheduled"] = int(
            slot["Class_Scheduled"]
        )

    return df


train_df = add_timetable_feature(pd.read_csv(TRAIN_FILE))

features = [
    "temperature",
    "humidity",
    "light",
    "co2",
    "humidityratio",
    "hour",
    "day_of_week",
    "class_scheduled",
]

X_train = train_df[features]
y_train = train_df["occupancy"].astype(int)

print("Training rows:", len(train_df))
print("Features:", features)
print("\nTimetable feature counts:")
print(train_df["class_scheduled"].value_counts().sort_index())
print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

test_data = []

for filename in TEST_FILES:
    df = add_timetable_feature(pd.read_csv(filename))
    test_data.append(
        (filename.name, df[features], df["occupancy"].astype(int))
    )

models = {
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(),
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
}

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

results = []
best_model = None
best_name = None
best_f1 = -1

for name, model in models.items():
    print(f"\nTraining {name}...")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start

    for test_name, X_test, y_test in test_data:
        start = time.perf_counter()
        pred = model.predict(X_test)
        prediction_seconds = time.perf_counter() - start

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(
            y_test, pred, zero_division=0
        )
        recall = recall_score(
            y_test, pred, zero_division=0
        )
        f1 = f1_score(y_test, pred, zero_division=0)

        print(f"\n{name} | {test_name}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, pred))

        results.append({
            "Model": name,
            "Test file": test_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Training time (s)": training_seconds,
            "Prediction time (s)": prediction_seconds,
        })

        if test_name == TEST_FILES[0].name and f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

print("\nMAJORITY-CLASS BASELINE")
for test_name, X_test, y_test in test_data:
    pred = baseline.predict(X_test)
    print(
        test_name,
        "| Accuracy:", round(accuracy_score(y_test, pred), 4),
        "| F1:", round(f1_score(y_test, pred, zero_division=0), 4),
    )

pd.DataFrame(results).to_csv(
    BASE / "timetable_model_results.csv", index=False
)

if best_model is not None:
    joblib.dump(
        best_model,
        MODELS / "timetable_occupancy_model.pkl",
    )
    (MODELS / "timetable_model_name.txt").write_text(
        best_name, encoding="utf-8"
    )

    print("\nSelected model:", best_name)
    print("Selection F1-score:", round(best_f1, 4))
    print("Saved timetable-aware model.")

print("\nResults saved to timetable_model_results.csv")
print(
    "NOTE: A101 timetable is illustrative; "
    "these results do not validate a real college schedule."
)
