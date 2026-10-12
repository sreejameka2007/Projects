
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

RESULT_FILES = {
    "Sensor-only": BASE_DIR / "model_results.csv",
    "Timetable-aware": BASE_DIR / "timetable_model_results.csv",
}

METRICS = ["Accuracy", "Precision", "Recall", "F1-score"]


def load_results(label, path):
    if not path.exists():
        print(f"WARNING: Results file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = {"Model", "Accuracy", "F1-score"}
    missing = required - set(df.columns)

    if missing:
        print(f"WARNING: {path.name} is missing columns: {missing}")
        print("Available columns:", list(df.columns))
        return pd.DataFrame()

    df["Experiment"] = label

    for column in METRICS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    return df


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / filename,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()


def plot_metric_comparison(results, metric):
    if metric not in results.columns:
        return

    data = results.dropna(subset=[metric])

    if data.empty:
        return

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=data,
        x="Model",
        y=metric,
        hue="Experiment",
        errorbar=None,
    )

    plt.title(f"Model Comparison — {metric}")
    plt.ylabel(metric)
    plt.xlabel("Machine Learning Model")

    if metric != "Prediction time (s)" and metric != "Training time (s)":
        plt.ylim(0, 1.05)

    save_plot(f"model_comparison_{metric.lower().replace(' ', '_').replace('-', '')}.png")


def main():
    frames = []

    for label, path in RESULT_FILES.items():
        df = load_results(label, path)
        if not df.empty:
            frames.append(df)

    if not frames:
        print(
            "No usable result files found. "
            "Check the CSV paths and column names."
        )
        return

    results = pd.concat(frames, ignore_index=True)

    print("\nMODEL EVALUATION RESULTS")
    print("=" * 70)
    print(results.to_string(index=False))

    results.to_csv(
        BASE_DIR / "reports" / "combined_model_results.csv",
        index=False,
    )

    # 1. Classification metrics
    for metric in METRICS:
        plot_metric_comparison(results, metric)

    # 2. Training and prediction time
    for metric in ["Training time (s)", "Prediction time (s)"]:
        if metric in results.columns:
            plot_metric_comparison(results, metric)

    # 3. Best model per experiment and test file
    if "Test file" in results.columns:
        summary = (
            results.sort_values("F1-score", ascending=False)
            .groupby(["Experiment", "Test file"], as_index=False)
            .first()
        )
    else:
        summary = (
            results.sort_values("F1-score", ascending=False)
            .groupby("Experiment", as_index=False)
            .first()
        )

    summary.to_csv(
        BASE_DIR / "reports" / "best_model_summary.csv",
        index=False,
    )

    print("\nBEST RESULTS BY EXPERIMENT AND TEST SET")
    print(summary.to_string(index=False))

    print("\nEvaluation graphs saved in:", FIGURES_DIR)
    print("Combined results saved in reports/combined_model_results.csv")
    print("Best-model summary saved in reports/best_model_summary.csv")

    print(
        "\nNOTE: These are comparisons of the existing experiments. "
        "Timetable-aware results use an illustrative schedule and "
        "must not be interpreted as proof of improvement on real campus data."
    )


if __name__ == "__main__":
    main()
