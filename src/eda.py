
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
DATA_PATH = Path("data/processed/cleaned_credit_risk.csv")
OUTPUT_DIR = Path("results/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")

print("=" * 50)
print("CREDIT RISK DATASET - EDA")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nLoan Status Distribution:")
print(df["loan_status"].value_counts())

print("\nLoan Status Percentage:")
print(
    df["loan_status"].value_counts(normalize=True) * 100
)

# 1. Loan Status Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="loan_status")
plt.title("Loan Default Distribution")
plt.xlabel("Loan Status (0 = Non-Default, 1 = Default)")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_loan_status_distribution.png")
plt.close()

# 2. Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="person_age", bins=30, kde=True)
plt.title("Applicant Age Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_age_distribution.png")
plt.close()

# 3. Income Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="person_income", bins=40, kde=True)
plt.title("Applicant Income Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_income_distribution.png")
plt.close()

# 4. Loan Amount Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="loan_amnt", bins=40, kde=True)
plt.title("Loan Amount Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_loan_amount_distribution.png")
plt.close()

# 5. Loan Grade vs Default Rate
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="loan_grade", y="loan_status")
plt.title("Default Rate by Loan Grade")
plt.ylabel("Average Default Rate")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_grade_vs_default.png")
plt.close()

# 6. Loan Intent vs Default Rate
plt.figure(figsize=(10, 5))
sns.barplot(
    data=df,
    x="loan_intent",
    y="loan_status"
)
plt.title("Default Rate by Loan Purpose")
plt.xticks(rotation=30)
plt.ylabel("Average Default Rate")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_intent_vs_default.png")
plt.close()

# 7. Home Ownership vs Default Rate
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="person_home_ownership",
    y="loan_status"
)
plt.title("Default Rate by Home Ownership")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_home_ownership_vs_default.png")
plt.close()

# 8. Correlation Heatmap
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_correlation_heatmap.png")
plt.close()

print("\nEDA completed successfully!")
print("Charts saved inside:", OUTPUT_DIR)
