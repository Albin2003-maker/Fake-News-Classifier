
import pandas as pd
from pathlib import Path

file_path = Path(__file__).parent / "WELFake_Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

if "label" in df.columns:
    print("\nLabel counts:")
    print(df["label"].value_counts(dropna=False))

print("\n========== LABEL EXAMPLES ==========")

for label in sorted(df["label"].unique()):
    print(f"\n--- LABEL {label} ---")

    examples = df.loc[df["label"] == label, "title"].dropna().head(5)

    for title in examples:
        print("-", title)
