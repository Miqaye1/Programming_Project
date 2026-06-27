import pandas as pd

df = pd.read_csv("../data/cleaned/cs-training-cleaned.csv")

df["TotalLatePayments"] = (
    df["NumberOfTime30-59DaysPastDueNotWorse"] +
    df["NumberOfTime60-89DaysPastDueNotWorse"] +
    df["NumberOfTimes90DaysLate"]
)

df["IncomePerDependent"] = (
    df["MonthlyIncome"] /
    (df["NumberOfDependents"] + 1)
)

df["HighUtilization"] = (
    df["RevolvingUtilizationOfUnsecuredLines"] > 0.8
).astype(int)

df["HighDebt"] = (
    df["DebtRatio"] > 1
).astype(int)

df["Senior"] = (
    df["age"] >= 65
).astype(int)

df.to_csv(
    "../data/cleaned/feature_engineered.csv",
    index=False
)