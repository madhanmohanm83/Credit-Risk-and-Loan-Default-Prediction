import json
import joblib
import pandas as pd

MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "data/features/preprocessor.pkl"
THRESHOLD_PATH = "results/models/best_threshold.json"


def load_prediction_assets():
    """Load trained model, preprocessor and threshold."""

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    with open(THRESHOLD_PATH, "r") as file:
        threshold_data = json.load(file)

    threshold = float(threshold_data["threshold"])

    return model, preprocessor, threshold


def predict_loan_risk(applicant_data):

    model, preprocessor, threshold = load_prediction_assets()

    input_df = pd.DataFrame([applicant_data])

    processed_data = pd.DataFrame(
    preprocessor.transform(input_df),
    columns=preprocessor.get_feature_names_out()
)

    default_probability = model.predict_proba(processed_data)[0][1]

    prediction = int(default_probability >= threshold)

    if default_probability < 0.20:
        risk_level = "Low"
    elif default_probability < 0.50:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "prediction": prediction,
        "prediction_label": (
            "Default Risk" if prediction == 1 else "No Default Risk"
        ),
        "default_probability": round(float(default_probability), 4),
        "risk_level": risk_level,
        "threshold": round(threshold, 4)
    }


if __name__ == "__main__":

    sample_applicant = {
        "person_age": 28,
        "person_income": 60000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_int_rate": 11.5,
        "loan_percent_income": 0.17,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 5
    }

    result = predict_loan_risk(sample_applicant)

    print("\n========== LOAN RISK PREDICTION ==========")

    for key, value in result.items():
        print(f"{key}: {value}")