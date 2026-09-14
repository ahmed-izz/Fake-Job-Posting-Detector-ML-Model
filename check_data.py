import pandas as pd

file_path = r"C:\Users\acer\Downloads\fake_job_postings.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nFraud label distribution:")
print(df["fraudulent"].value_counts())

print("\nFraud percentages:")
print((df["fraudulent"].value_counts(normalize=True) * 100).round(2))