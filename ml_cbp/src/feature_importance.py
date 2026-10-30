
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance
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


def load_data(filename):
    df = pd.read_csv(DATA_DIR / filename, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek

    for col in [
        "Temperature", "Humidity", "Light",
        "CO2", "HumidityRatio", TARGET
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + [TARGET])
    return df


train = load_data("datatraining (1).txt")
test = load_data("datatest (1).txt")

X_train = train[FEATURES]
y_train = train[TARGET].astype(int)

X_test = test[FEATURES]
y_test = test[TARGET].astype(int)

# Use a reproducible sample to keep computation manageable.
sample_size = min(2000, len(X_test))
X_sample = X_test.sample(
    n=sample_size, random_state=42
)
y_sample = y_test.loc[X_sample.index]

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42),
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

all_results = []

for name, model in models.items():
    print(f"Evaluating feature importance: {name}")

    model.fit(X_train, y_train)

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="f1",
        n_repeats=3,
        random_state=42,
        n_jobs=-1,
    )

    for feature, importance, std in zip(
        FEATURES,
        result.importances_mean,
        result.importances_std,
    ):
        all_results.append({
            "Model": name,
            "Feature": feature,
            "Importance": importance,
            "Standard deviation": std,
        })

results = pd.DataFrame(all_results)

results.to_csv(
    REPORTS_DIR / "feature_importance_results.csv",
    index=False,
)

# Create a separate plot for each model.
for name in models:
    subset = results[results["Model"] == name]
    subset = subset.sort_values("Importance")

    plt.figure(figsize=(8, 5))
    plt.barh(subset["Feature"], subset["Importance"])
    plt.axvline(0, linestyle="--")
    plt.xlabel("Decrease in F1-score after shuffling")
    plt.ylabel("Feature")
    plt.title(f"Permutation Feature Importance — {name}")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")
    plt.savefig(
        FIGURES_DIR / f"feature_importance_{safe_name}.png",
        dpi=300,
    )
    plt.close()

print("\nFeature importance analysis completed.")
print(f"CSV: {REPORTS_DIR / 'feature_importance_results.csv'}")
print(f"Charts saved in: {FIGURES_DIR}")
