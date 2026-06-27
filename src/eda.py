import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/cleaned/cs-training-cleaned.csv")

print(df.head())

print(df.shape)

print(df.info())

print(df.describe())

print(df["SeriousDlqin2yrs"].value_counts())

print(df["SeriousDlqin2yrs"].value_counts(normalize=True) * 100)

df["SeriousDlqin2yrs"].value_counts().plot(kind="bar")

plt.title("Target Variable Distribution")
plt.xlabel("SeriousDlqin2yrs")
plt.ylabel("Count")

plt.show()

numeric_cols = df.select_dtypes(include="number").columns

df[numeric_cols].hist(
    figsize=(18, 12),
    bins=30
)

plt.tight_layout()
plt.show()

for col in numeric_cols:

    plt.figure(figsize=(6,3))

    plt.boxplot(df[col])

    plt.title(col)

    plt.show()

corr = df.corr(numeric_only=True)

print(corr)

plt.figure(figsize=(12,10))

plt.imshow(corr)

plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.tight_layout()

plt.show()

print(
    corr["SeriousDlqin2yrs"]
    .sort_values(ascending=False)
)

plt.figure(figsize=(8,5))

plt.scatter(
    df["age"],
    df["SeriousDlqin2yrs"],
    alpha=0.2
)

plt.xlabel("Age")
plt.ylabel("Default")

plt.show()

plt.figure(figsize=(8,5))

plt.hist(df["MonthlyIncome"], bins=50)

plt.title("Monthly Income")

plt.show()

plt.figure(figsize=(8,5))

plt.hist(df["DebtRatio"], bins=50)

plt.title("Debt Ratio")

plt.show()

plt.figure(figsize=(8,5))

plt.hist(df["NumberOfOpenCreditLinesAndLoans"], bins=30)

plt.title("Open Credit Lines")

plt.show()

