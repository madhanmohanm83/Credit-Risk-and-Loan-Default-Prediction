
import pandas as pd
import numpy as np
import json

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

DATA_PATH = Path("data/processed/cleaned_credit_risk.csv")
OUTPUT_DIR = Path("data/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Separate features and target
X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# --------------------------------------------------
# 3. IDENTIFY COLUMN TYPES
# --------------------------------------------------

numeric_columns = X.select_dtypes(include="number").columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nNumeric columns:", numeric_columns)
print("\nCategorical columns:", categorical_columns)

# --------------------------------------------------
# 4. TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])

# --------------------------------------------------
# 5. NUMERICAL PIPELINE
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# --------------------------------------------------
# 6. CATEGORICAL PIPELINE
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# --------------------------------------------------
# 7. COMBINE PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ]
)

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using training-fitted rules
X_test_processed = preprocessor.transform(X_test)

# --------------------------------------------------
# 8. SAVE PROCESSED DATA
# --------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

X_train_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_test_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)

X_train_df.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test_df.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

# Save fitted preprocessing pipeline
import joblib

joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.pkl")

# Save feature names
with open(OUTPUT_DIR / "feature_names.json", "w") as file:
    json.dump(feature_names.tolist(), file, indent=4)

# --------------------------------------------------
# 9. FINAL REPORT
# --------------------------------------------------

print("\n========== PREPROCESSING RESULTS ==========")

print("Original feature count:", X.shape[1])
print("Processed feature count:", X_train_df.shape[1])

print("X_train shape:", X_train_df.shape)
print("X_test shape:", X_test_df.shape)

print("\nMissing values in training data:", X_train_df.isnull().sum().sum())
print("Missing values in testing data:", X_test_df.isnull().sum().sum())

print("\nTarget distribution in training data:")
print(y_train.value_counts())

print("\nTarget distribution in testing data:")
print(y_test.value_counts())

print("\nAll preprocessing files saved successfully!")
print("Output folder:", OUTPUT_DIR)
