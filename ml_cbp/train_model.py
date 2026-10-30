
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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_FILE = DATA_DIR / "datatraining (1).txt"
TEST_FILES = [
    DATA_DIR / "datatest (1).txt",
    DATA_DIR / "datatest2 (1).txt",
]

def load_data(path):
    df = pd.read_csv(path)

    # Clean column names and remove any saved row-index column.
    df.columns = [
        str(col).strip().strip('"').lower()
        for col in df.columns
    ]
    df = df.loc[
        :, ~df.columns.str.startswith("unnamed")
    ]

    required = {
        "date", "temperature", "humidity",
        "light", "co2", "humidityratio", "occupancy"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path.name}: {missing}. "
            f"Found: {df.columns.tolist()}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in [
        "temperature", "humidity", "light",
        "co2", "humidityratio", "occupancy"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive time features from the timestamp.
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek

    features = [
        "temperature", "humidity", "light",
        "co2", "humidityratio", "hour", "day_of_week"
    ]

    df = df.dropna(subset=features + ["occupancy"])
    df["occupancy"] = df["occupancy"].astype(int)

    return df, features


# Load the designated training dataset.
train_df, features = load_data(TRAIN_FILE)

print("\nTRAINING DATA")
print("-" * 45)
print("Rows:", len(train_df))
print("Features:", features)
print("Missing values in selected columns:")
print(train_df[features + ["occupancy"]].isnull().sum())
print("\nOccupancy distribution:")
print(train_df["occupancy"].value_counts().sort_index())

X_train = train_df[features]
y_train = train_df["occupancy"]

# The original UCI test files remain separate.
test_data = []
for path in TEST_FILES:
    df, test_features = load_data(path)
    test_data.append((path.name, df[test_features], df["occupancy"]))

# Compare every model using the same training and test data.
models = {
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(probability=True, random_state=42)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),
}

# Baseline: always predict the majority class in training data.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

results = []
best_model = None
best_model_name = None
best_f1 = -1

for name, model in models.items():
    print(f"\nTraining {name}...")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start

    # Evaluate on both official test files.
    for test_name, X_test, y_test in test_data:
        start = time.perf_counter()
        predictions = model.predict(X_test)
        prediction_seconds = time.perf_counter() - start

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(y_test, predictions, zero_division=0)

        print(f"\n{name} | {test_name}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))
        print(f"Training time: {training_seconds:.3f}s")
        print(f"Prediction time: {prediction_seconds:.4f}s")

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

        # Select using F1 on the first official test set.
        if test_name == TEST_FILES[0].name and f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

# Evaluate the baseline on the same test sets.
print("\nMAJORITY-CLASS BASELINE")
for test_name, X_test, y_test in test_data:
    predictions = baseline.predict(X_test)
    print(
        test_name,
        "| Accuracy:", round(accuracy_score(y_test, predictions), 4),
        "| F1:", round(f1_score(y_test, predictions, zero_division=0), 4)
    )

results_df = pd.DataFrame(results)
results_df.to_csv(BASE_DIR / "model_results.csv", index=False)

if best_model is not None:
    candidate_path = MODEL_DIR / "occupancy_model_candidate.pkl"

    joblib.dump(best_model, candidate_path)

    (MODEL_DIR / "model_name_candidate.txt").write_text(
        best_model_name,
        encoding="utf-8"
    )

    print("\nSELECTED MODEL:", best_model_name)
    print("Selection F1-score:", round(best_f1, 4))
    print("Saved candidate model to:", candidate_path)


print("\nSaved comparison results to model_results.csv")
print("Training and evaluation completed.")
