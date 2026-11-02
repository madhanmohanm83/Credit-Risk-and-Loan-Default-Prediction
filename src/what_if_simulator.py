
import pandas as pd
from src.prediction import predict_loan_risk


def run_what_if_simulation(applicant_data):
    """
    Compare the original loan risk with alternative scenarios.
    Each scenario changes one input at a time.
    """

    scenarios = {
        "Original Application": {},

        "Loan Amount Reduced by 20%": {
            "loan_amnt": applicant_data["loan_amnt"] * 0.80
        },

        "Interest Rate Reduced by 2%": {
            "loan_int_rate": max(
                1.0, applicant_data["loan_int_rate"] - 2.0
            )
        },

        "Income Increased by 20%": {
            "person_income": applicant_data["person_income"] * 1.20
        },

        "Loan Amount Reduced by 20% + Interest Rate Reduced by 2%": {
            "loan_amnt": applicant_data["loan_amnt"] * 0.80,
            "loan_int_rate": max(
                1.0, applicant_data["loan_int_rate"] - 2.0
            )
        }
    }

    results = []

    for scenario_name, changes in scenarios.items():

        modified_applicant = applicant_data.copy()
        modified_applicant.update(changes)

        # Recalculate loan-to-income ratio
        modified_applicant["loan_percent_income"] = (
            modified_applicant["loan_amnt"]
            / modified_applicant["person_income"]
        )

        prediction = predict_loan_risk(modified_applicant)

        results.append({
            "Scenario": scenario_name,
            "Loan Amount": round(
                modified_applicant["loan_amnt"], 2
            ),
            "Interest Rate": round(
                modified_applicant["loan_int_rate"], 2
            ),
            "Annual Income": round(
                modified_applicant["person_income"], 2
            ),
            "Default Probability (%)": round(
                prediction["default_probability"] * 100, 2
            ),
            "Risk Level": prediction["risk_level"],
            "Prediction": prediction["prediction_label"]
        })

    return pd.DataFrame(results)


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
        "loan_percent_income": 10000 / 60000,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 5
    }

    simulation_results = run_what_if_simulation(
        sample_applicant
    )

    print("\n========== WHAT-IF LOAN SIMULATOR ==========\n")
    print(simulation_results.to_string(index=False))

    simulation_results.to_csv(
        "results/explainability/what_if_results.csv",
        index=False
    )

    print("\nSimulation results saved successfully!")
