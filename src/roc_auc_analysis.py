
from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    auc,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio",
    "hour",
    "day_of_week",
]

TARGET = "Occupancy"


def load_dataset(filename):
    path = DATA_DIR / filename
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek

    for column in [
        "Temperature", "Humidity", "Light",
        "CO2", "HumidityRatio", TARGET
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURES + [TARGET])
    return df


models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    ),
}

train = load_dataset("datatraining (1).txt")
test_files = [
    "datatest (1).txt",
    "datatest2 (1).txt",
]

X_train = train[FEATURES]
y_train = train[TARGET].astype(int)

results = []
plt.figure(figsize=(10, 7))

for name, model in models.items():
    print(f"Training {name}...")

    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)

    for filename in test_files:
        test = load_dataset(filename)
        X_test = test[FEATURES]
        y_test = test[TARGET].astype(int)

        # Probability that the room is occupied (class 1)
        scores = fitted_model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_value = roc_auc_score(y_test, scores)

        results.append({
            "Model": name,
            "Test file": filename,
            "ROC-AUC": auc_value,
        })

        plt.plot(
            fpr,
            tpr,
            label=f"{name} | {filename}: AUC={auc_value:.3f}"
        )

        print(
            f"  {filename}: ROC-AUC = {auc_value:.4f}"
        )

plt.plot(
    [0, 1], [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Classroom Occupancy Models")
plt.legend(fontsize=7, loc="lower right")
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "roc_auc_comparison.png",
    dpi=300
)
plt.close()

results_df = pd.DataFrame(results)
results_df.to_csv(
    REPORTS_DIR / "roc_auc_results.csv",
    index=False
)

print("\nROC-AUC analysis completed.")
print(f"Results: {REPORTS_DIR / 'roc_auc_results.csv'}")
print(f"Chart: {FIGURES_DIR / 'roc_auc_comparison.png'}")
