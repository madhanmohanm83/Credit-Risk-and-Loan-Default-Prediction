
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

DATA_DIR = Path("data/features")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results/models")

RESULT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# --------------------------------------------------
# 2. LOAD PREPROCESSED DATA
# --------------------------------------------------

X_train_full = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train_full = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print("Full training data:", X_train_full.shape)
print("Final test data:", X_test.shape)

# --------------------------------------------------
# 3. CREATE TRAINING AND VALIDATION SETS
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Final test records:", len(X_test))

# --------------------------------------------------
# 4. TRAIN RANDOM FOREST
# --------------------------------------------------

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# --------------------------------------------------
# 5. VALIDATION THRESHOLD TUNING
# --------------------------------------------------

val_probabilities = rf_model.predict_proba(X_val)[:, 1]

precision_values, recall_values, thresholds = (
    precision_recall_curve(y_val, val_probabilities)
)

# Choose threshold with best F1 score
f1_values = (
    2 * precision_values[:-1] * recall_values[:-1]
    / (precision_values[:-1] + recall_values[:-1] + 1e-10)
)

best_index = np.argmax(f1_values)
best_threshold = float(thresholds[best_index])

print("\nBest validation threshold:", round(best_threshold, 4))
print("Validation F1 score:", round(f1_values[best_index], 4))

# Save threshold
with open(RESULT_DIR / "best_threshold.json", "w") as file:
    json.dump(
        {"threshold": best_threshold},
        file,
        indent=4
    )

# --------------------------------------------------
# 6. FINAL TEST EVALUATION
# --------------------------------------------------

test_probabilities = rf_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions, zero_division=0)
recall = recall_score(y_test, test_predictions, zero_division=0)
f1 = f1_score(y_test, test_predictions, zero_division=0)
roc_auc = roc_auc_score(y_test, test_probabilities)

print("\n========== FINAL TEST RESULTS ==========")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC AUC:", round(roc_auc, 4))

# --------------------------------------------------
# 7. CLASSIFICATION REPORT
# --------------------------------------------------

report = classification_report(
    y_test,
    test_predictions,
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open(RESULT_DIR / "final_classification_report.txt", "w") as file:
    file.write(report)

# --------------------------------------------------
# 8. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_test, test_predictions)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-Default", "Default"],
    yticklabels=["Non-Default", "Default"]
)

plt.title("Random Forest - Final Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "final_confusion_matrix.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 9. FALSE POSITIVE AND FALSE NEGATIVE ANALYSIS
# --------------------------------------------------

tn, fp, fn, tp = cm.ravel()

error_analysis = {
    "True Negatives": int(tn),
    "False Positives": int(fp),
    "False Negatives": int(fn),
    "True Positives": int(tp)
}

print("\n========== ERROR ANALYSIS ==========")

for key, value in error_analysis.items():
    print(key + ":", value)

with open(RESULT_DIR / "error_analysis.json", "w") as file:
    json.dump(error_analysis, file, indent=4)

# --------------------------------------------------
# 10. SAVE FINAL MODEL
# --------------------------------------------------

joblib.dump(
    rf_model,
    MODEL_DIR / "best_model.pkl"
)

# --------------------------------------------------
# 11. SAVE FINAL METRICS
# --------------------------------------------------

final_metrics = {
    "model": "Random Forest",
    "threshold": best_threshold,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc
}

with open(RESULT_DIR / "final_metrics.json", "w") as file:
    json.dump(final_metrics, file, indent=4)

print("\nFinal model and evaluation results saved.")
print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
