
import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------

DATA_DIR = Path("data/features")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# 2. LOAD TRAINING AND TESTING DATA
# ---------------------------------------

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ---------------------------------------
# 3. DEFINE MACHINE LEARNING MODELS
# ---------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=10,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
}

# ---------------------------------------
# 4. TRAIN AND EVALUATE MODELS
# ---------------------------------------

results = []
trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    trained_models[name] = model

    # Save individual model
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, MODEL_DIR / filename)

    # Save classification report
    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    report_file = (
        name.lower().replace(" ", "_")
        + "_report.txt"
    )

    with open(RESULT_DIR / report_file, "w") as file:
        file.write(report)

    # Save confusion matrix
    cm = confusion_matrix(y_test, predictions)

    cm_file = (
        name.lower().replace(" ", "_")
        + "_confusion_matrix.csv"
    )

    pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    ).to_csv(RESULT_DIR / cm_file)

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(roc_auc, 4))

# ---------------------------------------
# 5. COMPARE MODEL RESULTS
# ---------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    RESULT_DIR / "model_comparison.csv",
    index=False
)

# ---------------------------------------
# 6. SELECT BEST MODEL
# ---------------------------------------

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.pkl"
)

with open(RESULT_DIR / "best_model.json", "w") as file:
    json.dump(
        {
            "best_model": best_model_name,
            "selection_metric": "F1 Score"
        },
        file,
        indent=4
    )

print("\nBest Model:", best_model_name)
print("Saved to:", MODEL_DIR / "best_model.pkl")
print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
