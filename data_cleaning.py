import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/cs-training.csv")

# First 5 rows
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# Shape
print("\n========== SHAPE ==========")
print(df.shape)

# Columns
print("\n========== COLUMNS ==========")
print(df.columns.tolist())

# Data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# Dataset info
print("\n========== INFO ==========")
df.info()

# Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Duplicate rows
print("\n========== DUPLICATES ==========")
print(df.duplicated().sum())

# Statistics
print("\n========== STATISTICS ==========")
print(df.describe())