
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "datatraining (1).txt",
    "datatest (1).txt",
    "datatest2 (1).txt",
]

sns.set_theme(style="whitegrid")


def load_dataset(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    # Standardize column names.
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.strip('"')
        .str.lower()
    )

    # Remove an exported row-number column if present.
    df = df.drop(columns=["unnamed: 0"], errors="ignore")

    # Convert known fields to their appropriate types.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = [
        "temperature",
        "humidity",
        "light",
        "co2",
        "humidityratio",
        "occupancy",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def audit_dataset(filename, df):
    missing = df.isna().sum()

    return {
        "Dataset": filename,
        "Rows": len(df),
        "Columns": len(df.columns),
        "Duplicate rows": int(df.duplicated().sum()),
        "Missing cells": int(missing.sum()),
        "Missing date values": (
            int(df["date"].isna().sum())
            if "date" in df.columns else None
        ),
        "Invalid occupancy labels": (
            int((~df["occupancy"].isin([0, 1])).sum())
            if "occupancy" in df.columns else None
        ),
    }


def save_missing_value_chart(df, filename):
    missing = df.isna().sum()
    missing = missing[missing > 0]

    plt.figure(figsize=(9, 4))

    if missing.empty:
        plt.text(
            0.5, 0.5,
            "No missing values found",
            ha="center", va="center", fontsize=14,
        )
        plt.axis("off")
    else:
        missing.sort_values().plot(kind="bar")
        plt.ylabel("Number of missing values")
        plt.xlabel("Column")
        plt.title(f"Missing Values — {filename}")

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"{Path(filename).stem}_missing_values.png",
        dpi=150,
    )
    plt.close()


def save_occupancy_chart(df, filename):
    if "occupancy" not in df.columns:
        return

    counts = df["occupancy"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.xticks([0, 1], ["Not Occupied (0)", "Occupied (1)"], rotation=0)
    plt.ylabel("Number of records")
    plt.title(f"Occupancy Distribution — {filename}")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"{Path(filename).stem}_occupancy.png",
        dpi=150,
    )
    plt.close()


def main():
    reports = []

    for filename in FILES:
        print(f"\nInspecting: {filename}")

        df = load_dataset(filename)
        report = audit_dataset(filename, df)
        reports.append(report)

        print(pd.Series(report).to_string())
        print("\nData types:")
        print(df.dtypes.to_string())

        print("\nMissing values by column:")
        print(df.isna().sum().to_string())

        save_missing_value_chart(df, filename)
        save_occupancy_chart(df, filename)

    report_df = pd.DataFrame(reports)
    report_path = REPORT_DIR / "data_quality_report.csv"
    report_df.to_csv(report_path, index=False)

    print("\n" + "=" * 55)
    print("DATASET AUDIT COMPLETED")
    print("=" * 55)
    print(report_df.to_string(index=False))
    print(f"\nQuality report: {report_path}")
    print(f"Graphs saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
