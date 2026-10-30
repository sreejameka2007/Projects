
from pathlib import Path
import time

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
REPORTS = BASE / "reports"
FIGURES = REPORTS / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

TRAIN_FILE = DATA / "datatraining (1).txt"
TEST_FILES = [
    DATA / "datatest (1).txt",
    DATA / "datatest2 (1).txt",
]

SENSOR_FEATURES = [
    "temperature",
    "humidity",
    "light",
    "co2",
    "humidityratio",
    "hour",
    "day_of_week",
]

TIMETABLE_FEATURES = SENSOR_FEATURES + ["class_scheduled"]


def load_data(path):
    df = pd.read_csv(path)
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.strip('"')
        .str.lower()
    )
    df = df.drop(columns=["unnamed: 0"], errors="ignore")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric = [
        "temperature", "humidity", "light",
        "co2", "humidityratio", "occupancy",
    ]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"] + numeric).copy()
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek

    return df


def build_models():
    return {
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC()),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
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


def save_confusion_matrix(y_true, y_pred, name):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Occupied", "Occupied"],
        yticklabels=["Not Occupied", "Occupied"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()

    safe_name = "".join(
        c.lower() if c.isalnum() else "_"
        for c in name
    )
    plt.savefig(
        FIGURES / f"confusion_matrix_{safe_name}.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()


def main():
    train = load_data(TRAIN_FILE)
    X_train = train[SENSOR_FEATURES]
    y_train = train["occupancy"].astype(int)

    test_sets = []

    for path in TEST_FILES:
        test = load_data(path)
        test_sets.append((
            path.stem,
            test[SENSOR_FEATURES],
            test["occupancy"].astype(int),
            test,
        ))

    models = build_models()

    # A simple baseline that always predicts the most frequent training class.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    all_metrics = []
    all_predictions = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - start

        for test_name, X_test, y_test, test_df in test_sets:
            start = time.perf_counter()
            predictions = model.predict(X_test)
            prediction_time = time.perf_counter() - start

            metrics = {
                "Model": name,
                "Test file": test_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "F1-score": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "Training time (s)": training_time,
                "Prediction time (s)": prediction_time,
            }
            all_metrics.append(metrics)

            prefix = f"{name}_{test_name}"
            save_confusion_matrix(y_test, predictions, prefix)

            print(f"\n{name} | {test_name}")
            print(classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Not Occupied", "Occupied"],
                zero_division=0,
            ))

            prediction_frame = pd.DataFrame({
                "Model": name,
                "Test file": test_name,
                "Timestamp": test_df["date"].astype(str).values,
                "Actual occupancy": y_test.values,
                "Predicted occupancy": predictions,
            })
            all_predictions.append(prediction_frame)

    # Baseline evaluation
    for test_name, X_test, y_test, test_df in test_sets:
        predictions = baseline.predict(X_test)

        all_metrics.append({
            "Model": "Majority Baseline",
            "Test file": test_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "Training time (s)": None,
            "Prediction time (s)": None,
        })

        save_confusion_matrix(
            y_test, predictions, f"Majority_Baseline_{test_name}"
        )

        all_predictions.append(pd.DataFrame({
            "Model": "Majority Baseline",
            "Test file": test_name,
            "Timestamp": test_df["date"].astype(str).values,
            "Actual occupancy": y_test.values,
            "Predicted occupancy": predictions,
        }))

    pd.DataFrame(all_metrics).to_csv(
        REPORTS / "diagnostic_metrics.csv", index=False
    )

    pd.concat(all_predictions, ignore_index=True).to_csv(
        REPORTS / "test_predictions.csv", index=False
    )

    print("\nDiagnostics completed.")
    print("Metrics: reports/diagnostic_metrics.csv")
    print("Predictions: reports/test_predictions.csv")
    print("Confusion matrices: reports/figures/")

    print(
        "\nNote: these diagnostics evaluate sensor/time features. "
        "They do not establish real-world timetable-aware performance."
    )


if __name__ == "__main__":
    main()
