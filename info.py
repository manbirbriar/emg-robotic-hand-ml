from pathlib import Path

import pandas as pd


data_file = Path(__file__).with_name("master_emg_data.csv")
data = pd.read_csv(data_file)

print(f"Dataset: {data_file.name}")
print(f"Rows: {data.shape[0]:,}")
print(f"Columns: {data.shape[1]}")

print("\nColumn data types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isna().sum())

print("\nLabel distribution:")
print(data["Label"].value_counts().to_string())

print("\nNumeric summary:")
print(data.describe().round(3).to_string())
