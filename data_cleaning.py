import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/Topic2_credit_assignment_dataset.csv")

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

missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent.sort_values(ascending=False))

df["MonthlyIncome"] = df["MonthlyIncome"].fillna(
    df["MonthlyIncome"].median()
)

df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
    df["NumberOfDependents"].median()
)
print(df.isnull().sum())

print(df[df["age"] <= 0])

for col in df.select_dtypes(include="number"):
    print(col)
    print(df[col].describe())

df.to_csv(
    "data/cleaned/cs-training-cleaned.csv",
    index=False
)
print("\n========== IMPOSSIBLE VALUES ==========")

print("Age <= 0:", len(df[df["age"] <= 0]))
print("MonthlyIncome < 0:", len(df[df["MonthlyIncome"] < 0]))
print("DebtRatio < 0:", len(df[df["DebtRatio"] < 0]))
print("NumberOfDependents < 0:", len(df[df["NumberOfDependents"] < 0]))
print("RevolvingUtilizationOfUnsecuredLines < 0:",
      len(df[df["RevolvingUtilizationOfUnsecuredLines"] < 0]))

print("Rows before:", len(df))

df = df[df["age"] > 0]

print("Rows after:", len(df))

print(df["age"].describe())

print("\nOldest people:")
print(df.nlargest(10, "age")[["age"]])

print(df.columns.tolist())

print("\n========== OUTLIER COUNT ==========")

for col in df.select_dtypes(include="number"):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((df[col] < lower) | (df[col] > upper)).sum()

    print(f"{col}: {outliers}")

df.drop(columns=["Unnamed: 0"], inplace=True)
print(df.columns)

cols = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate"
]

for col in cols:
    print("\n", col)
    print(df[col].value_counts().head(20))

print(df["age"].describe())

print(df.nlargest(10, "age")[["age"]])

cols = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate"
]

for col in cols:
    print("\n", col)
    print(df[col].value_counts().head(20))

df.to_csv(
    "data/cleaned/cs-training-cleaned.csv",
    index=False
)