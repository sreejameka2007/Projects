
from pathlib import Path
import pandas as pd

# Locate the data folder relative to this Python file
DATA_DIR = Path(__file__).parent / "data"

files = [
    "datatraining (1).txt",
    "datatest (1).txt",
    "datatest2 (1).txt",
]

for filename in files:
    file_path = DATA_DIR / filename

    print("\n" + "=" * 55)
    print(f"FILE: {filename}")
    print("=" * 55)

    if not file_path.exists():
        print("ERROR: File not found:", file_path)
        continue

    df = pd.read_csv(file_path)

    print("\nDataset shape:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    if "Occupancy" in df.columns:
        print("\nOccupancy distribution:")
        print(df["Occupancy"].value_counts().sort_index())

    print("\nData types:")
    print(df.dtypes)

print("\nDataset inspection completed.")
