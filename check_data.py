import pandas as pd

# Load the dataset
file_path = "data/news-article-categories.csv"

df = pd.read_csv(file_path)

# Basic information
print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFORMATION =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== UNIQUE VALUES IN EACH COLUMN =====")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].nunique())