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
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

for col in df.columns:
    print("\n" + "=" * 60)
    print(f"Column: {col}")
    print(f"Data type: {df[col].dtype}")
    print(f"Missing values: {df[col].isnull().sum()}")
    print(f"Unique values: {df[col].nunique()}")

    print("\nTop values:")
    print(df[col].value_counts(dropna=False))

for col in df.select_dtypes(include="object"):
    print(f"\n{col}")
    print(df[col].unique())

# =====================================
# DATA CLEANING
# =====================================

print("\n========== STARTING DATA CLEANING ==========")

duplicates_before = df.duplicated().sum()
print(f"Duplicates before: {duplicates_before}")

df = df.drop_duplicates()

duplicates_after = df.duplicated().sum()
print(f"Duplicates after: {duplicates_after}")

for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip()

for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.title()

df.replace(
    ["?", "NA", "N/A", "Unknown", "None", ""],
    pd.NA,
    inplace=True
)

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())