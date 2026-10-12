
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

FILES = {
    "Training": "datatraining (1).txt",
    "Test 1": "datatest (1).txt",
    "Test 2": "datatest2 (1).txt",
}

SENSOR_COLUMNS = [
    "temperature",
    "humidity",
    "light",
    "co2",
    "humidityratio",
]


def load_data(filename):
    df = pd.read_csv(DATA_DIR / filename)

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.strip('"')
        .str.lower()
    )

    df = df.drop(columns=["unnamed: 0"], errors="ignore")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for column in SENSOR_COLUMNS + ["occupancy"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(
        subset=["date"] + SENSOR_COLUMNS + ["occupancy"]
    ).copy()

    return df


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / filename,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()


def main():
    datasets = {
        name: load_data(filename)
        for name, filename in FILES.items()
    }

    # 1. Dataset size comparison
    sizes = pd.Series({
        name: len(df) for name, df in datasets.items()
    })

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=sizes.index, y=sizes.values)
    ax.bar_label(ax.containers[0], fmt="%.0f")
    plt.title("Dataset Size Comparison")
    plt.xlabel("Dataset")
    plt.ylabel("Number of records")
    save_plot("eda_dataset_sizes.png")

    # 2. Occupancy distribution
    occupancy_frames = []

    for name, df in datasets.items():
        counts = (
            df["occupancy"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )

        for label, count in counts.items():
            occupancy_frames.append({
                "Dataset": name,
                "Occupancy": (
                    "Not Occupied" if label == 0 else "Occupied"
                ),
                "Records": int(count),
            })

    occupancy_df = pd.DataFrame(occupancy_frames)

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=occupancy_df,
        x="Dataset",
        y="Records",
        hue="Occupancy",
    )
    plt.title("Occupied vs Not Occupied Records")
    plt.ylabel("Number of records")
    save_plot("eda_occupancy_comparison.png")

    # 3. Sensor distributions
    training = datasets["Training"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, column in enumerate(SENSOR_COLUMNS):
        sns.histplot(
            data=training,
            x=column,
            hue="occupancy",
            bins=35,
            element="step",
            stat="density",
            common_norm=False,
            ax=axes[i],
        )
        axes[i].set_title(
            f"{column.title()} Distribution by Occupancy"
        )

    axes[-1].axis("off")
    fig.suptitle(
        "Environmental Sensor Distributions — Training Dataset",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "eda_sensor_distributions.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 4. Box plots for possible outliers
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, column in enumerate(SENSOR_COLUMNS):
        sns.boxplot(
            data=training,
            x="occupancy",
            y=column,
            ax=axes[i],
        )
        axes[i].set_title(f"{column.title()} by Occupancy")
        axes[i].set_xlabel("Occupancy (0 = No, 1 = Yes)")

    axes[-1].axis("off")
    fig.suptitle("Sensor Value and Outlier Exploration", fontsize=15)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "eda_sensor_boxplots.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 5. Correlation heatmap
    plt.figure(figsize=(9, 7))

    correlation_columns = SENSOR_COLUMNS + ["occupancy"]
    correlation = training[correlation_columns].corr()

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
    )

    plt.title("Sensor and Occupancy Correlation Heatmap")
    save_plot("eda_correlation_heatmap.png")

    # 6. Sensor trends over time (sampled for readability)
    sampled = training.iloc[::10].copy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for column, ax in zip(
        ["temperature", "light", "co2"], axes
    ):
        ax.plot(
            sampled["date"],
            sampled[column],
            linewidth=1,
        )
        ax.set_ylabel(column.title())
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestamp")
    fig.suptitle("Environmental Sensor Trends Over Time")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "eda_sensor_time_trends.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 7. Summary statistics
    summary = training[
        SENSOR_COLUMNS
    ].describe().T

    summary.to_csv(
        BASE_DIR / "reports" / "sensor_summary_statistics.csv"
    )

    print("\nEDA completed successfully.")
    print("\nTraining dataset summary:")
    print(summary.round(3).to_string())

    print(f"\nGraphs saved in: {FIGURES_DIR}")
    print(
        "Summary saved to reports/sensor_summary_statistics.csv"
    )


if __name__ == "__main__":
    main()
