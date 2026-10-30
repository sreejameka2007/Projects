
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RESULT_FILE = REPORTS_DIR / "diagnostic_metrics.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["Accuracy", "Precision", "Recall", "F1-score"]


def save_chart(fig, filename):
    path = FIGURES_DIR / filename
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved: {path}")


def main():
    if not RESULT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {RESULT_FILE}. "
            "Run python src/model_diagnostics.py first."
        )

    df = pd.read_csv(RESULT_FILE)
    df.columns = df.columns.str.strip()

    required = {
        "Model", "Test file", "Accuracy",
        "Precision", "Recall", "F1-score",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in diagnostic_metrics.csv: {missing}"
        )

    for column in METRICS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Model", "Test file"] + METRICS)

    if df.empty:
        raise ValueError("No valid evaluation results were found.")

    print("\nModels and test datasets found:")
    print(df[["Model", "Test file"]].to_string(index=False))

    # 1. Interactive classification metric comparison.
    metric_long = df.melt(
        id_vars=["Model", "Test file"],
        value_vars=METRICS,
        var_name="Metric",
        value_name="Score",
    )

    fig = px.bar(
        metric_long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        facet_col="Test file",
        title="ML Model Comparison — Classification Metrics",
        hover_data={"Score": ":.4f"},
    )
    fig.update_yaxes(range=[0, 1.05], title="Score")
    fig.update_xaxes(title="Model")
    fig.update_layout(
        template="plotly_white",
        legend_title="Metric",
    )
    save_chart(fig, "interactive_metric_comparison.html")

    # 2. F1-score ranking for each test dataset.
    fig = px.bar(
        df.sort_values("F1-score", ascending=True),
        x="F1-score",
        y="Model",
        color="Model",
        facet_col="Test file",
        orientation="h",
        text="F1-score",
        title="Model Ranking by F1-score",
        hover_data=["Accuracy", "Precision", "Recall"],
    )
    fig.update_xaxes(range=[0, 1.05], title="F1-score")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
    )
    save_chart(fig, "interactive_f1_ranking.html")

    # 3. Training-time comparison, if available.
    if "Training time (s)" in df.columns:
        training = df.dropna(subset=["Training time (s)"])

        if not training.empty:
            fig = px.bar(
                training.drop_duplicates(
                    subset=["Model", "Test file"]
                ),
                x="Model",
                y="Training time (s)",
                color="Model",
                facet_col="Test file",
                title="Model Training Time Comparison",
                hover_data=["F1-score"],
            )
            fig.update_layout(
                template="plotly_white",
                showlegend=False,
            )
            save_chart(fig, "interactive_training_time.html")

    # 4. Prediction-time comparison, if available.
    if "Prediction time (s)" in df.columns:
        prediction = df.dropna(subset=["Prediction time (s)"])

        if not prediction.empty:
            fig = px.bar(
                prediction,
                x="Model",
                y="Prediction time (s)",
                color="Model",
                facet_col="Test file",
                title="Model Prediction Time Comparison",
                hover_data=["F1-score"],
            )
            fig.update_layout(
                template="plotly_white",
                showlegend=False,
            )
            save_chart(fig, "interactive_prediction_time.html")

    # 5. Downloadable consolidated evaluation table.
    output_csv = REPORTS_DIR / "interactive_comparison_data.csv"
    df.to_csv(output_csv, index=False)

    print("\nVisualization generation completed.")
    print(f"Comparison data saved to: {output_csv}")
    print(f"Interactive graphs saved in: {FIGURES_DIR}")
    print(
        "\nNote: these charts use the diagnostic evaluation results. "
        "They do not establish that the illustrative timetable improves "
        "real-world occupancy prediction."
    )


if __name__ == "__main__":
    main()
