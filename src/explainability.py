import os
import json
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
MODEL_PATH = "models/best_model.pkl"
X_TRAIN_PATH = "data/features/X_train.csv"
X_TEST_PATH = "data/features/X_test.csv"
OUTPUT_DIR = "results/explainability"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model and datasets...")

# Load model and data
model = joblib.load(MODEL_PATH)
X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)

# Use a sample to keep SHAP computation manageable
background = X_train.sample(
    n=min(100, len(X_train)),
    random_state=42
)

sample_data = X_test.sample(
    n=min(200, len(X_test)),
    random_state=42
)

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_data)

# Handle binary classification output
if isinstance(shap_values, list):
    values = shap_values[1]
elif len(shap_values.shape) == 3:
    values = shap_values[:, :, 1]
else:
    values = shap_values

# Global feature importance
importance = pd.DataFrame({
    "feature": sample_data.columns,
    "mean_absolute_shap": abs(values).mean(axis=0)
})

importance = importance.sort_values(
    by="mean_absolute_shap",
    ascending=False
)

importance.to_csv(
    f"{OUTPUT_DIR}/feature_importance.csv",
    index=False
)

# Save SHAP summary plot
plt.figure()
shap.summary_plot(
    values,
    sample_data,
    show=False
)
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/shap_summary.png",
    bbox_inches="tight"
)
plt.close()

# Save feature importance chart
plt.figure(figsize=(10, 7))
top_features = importance.head(15).sort_values(
    "mean_absolute_shap"
)

plt.barh(
    top_features["feature"],
    top_features["mean_absolute_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.title("Top 15 Features Influencing Loan Default Predictions")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
plt.close()

print("\nSHAP Explainability completed!")
print("\nTop 10 important features:")
print(importance.head(10).to_string(index=False))

print(f"\nResults saved in: {OUTPUT_DIR}")