
import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

from src.prediction import predict_loan_risk
from src.what_if_simulator import run_what_if_simulation


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

METRICS_PATH = os.path.join(
    BASE_DIR, "results", "models", "final_metrics.json"
)

COMPARISON_PATH = os.path.join(
    BASE_DIR, "results", "models", "model_comparison.csv"
)

CONFUSION_MATRIX_PATH = os.path.join(
    BASE_DIR, "results", "models", "final_confusion_matrix.png"
)

FEATURE_PATH = os.path.join(
    BASE_DIR, "results", "explainability", "feature_importance.csv"
)

FEATURE_IMAGE_PATH = os.path.join(
    BASE_DIR, "results", "explainability", "feature_importance.png"
)

SHAP_SUMMARY_PATH = os.path.join(
    BASE_DIR, "results", "explainability", "shap_summary.png"
)


# ============================================================
# 3. CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>

/* Application background */
.stApp {
    background: #F5F7FB;
}

/* Main content */
[data-testid="stAppViewContainer"] {
    background: #F5F7FB;
}

[data-testid="stAppViewContainer"] .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Main headings */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 {
    color: #172B4D !important;
    font-weight: 700;
}

/* Main text */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label {
    color: #344563;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #13213B;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #FFFFFF !important;
}

/* Sidebar radio labels */
[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #FFFFFF !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #64748B !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #172B4D !important;
}

/* Text and number inputs */
[data-testid="stAppViewContainer"] input {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    -webkit-text-fill-color: #172B4D !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

/* Select boxes */
[data-testid="stAppViewContainer"] [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    border-color: #CBD5E1 !important;
}

[data-testid="stAppViewContainer"] [data-baseweb="select"] span {
    color: #172B4D !important;
}

/* Form containers */
[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 22px;
}

/* Buttons */
[data-testid="stAppViewContainer"] .stButton button,
[data-testid="stAppViewContainer"] .stFormSubmitButton button,
[data-testid="stAppViewContainer"] .stDownloadButton button {
    background-color: #173B70;
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    min-height: 42px;
}

[data-testid="stAppViewContainer"] .stButton button:hover,
[data-testid="stAppViewContainer"] .stFormSubmitButton button:hover,
[data-testid="stAppViewContainer"] .stDownloadButton button:hover {
    background-color: #24579B;
    color: #FFFFFF !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #DCE3ED;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def load_json(path):
    """Load JSON file safely."""
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        st.warning(f"Could not read {os.path.basename(path)}: {error}")
        return {}


def load_csv(path):
    """Load CSV file safely."""
    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.warning(f"Could not read {os.path.basename(path)}: {error}")
        return None


def show_metric(label, value):
    """Display a metric safely."""
    try:
        formatted = f"{float(value):.3f}"
    except (TypeError, ValueError):
        formatted = "N/A"

    st.metric(label, formatted)


def page_header(title, description):
    """Display consistent page heading."""
    st.title(title)
    st.caption(description)
    st.divider()


def get_probability_column(data):
    """Find probability column in What-If results."""
    for column in data.columns:
        if "probability" in column.lower():
            return column
    return None


def get_scenario_column(data):
    """Find scenario column in What-If results."""
    for column in data.columns:
        if "scenario" in column.lower():
            return column
    return None


def get_model_column(data):
    """Find model-name column in comparison results."""
    for column in data.columns:
        if any(word in column.lower() for word in ["model", "algorithm", "classifier"]):
            return column
    return data.columns[0]


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("# 💳 CREDIT RISK")
    st.markdown("### LOAN DEFAULT INTELLIGENCE")

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Dashboard Overview",
            "Loan Risk Prediction",
            "Model Performance",
            "Risk Explainability",
            "What-If Loan Simulator"
        ],
        key="selected_page"
    )

    st.divider()

    st.caption("MACHINE LEARNING POWERED")
    st.caption("CREDIT RISK ANALYTICS")

    st.divider()

    st.caption("Credit Risk & Loan Default Prediction")
    st.caption("Machine Learning Project")


# ============================================================
# 6. DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard Overview":

    page_header(
        "Credit Risk Intelligence Dashboard",
        "Machine Learning Powered Loan Default Prediction and Risk Analytics"
    )

    metrics = load_json(METRICS_PATH)

    st.subheader("Model Performance at a Glance")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        show_metric("Accuracy", metrics.get("accuracy"))

    with col2:
        show_metric("Precision", metrics.get("precision"))

    with col3:
        show_metric("Recall", metrics.get("recall"))

    with col4:
        show_metric("F1 Score", metrics.get("f1_score"))

    with col5:
        show_metric("ROC-AUC", metrics.get("roc_auc"))

    st.divider()

    st.subheader("Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            """
            **Project Objective**

            Identify potential loan defaults using applicant
            financial information, credit history and loan
            characteristics.
            """
        )

    with col2:
        st.success(
            """
            **Selected Model**

            Random Forest Classifier

            The trained model estimates default probability
            and predicts the applicant's default risk.
            """
        )

    st.subheader("Machine Learning Workflow")

    workflow = [
        "Data Collection",
        "Data Cleaning",
        "Exploratory Data Analysis",
        "Preprocessing",
        "Model Training",
        "Model Evaluation",
        "SHAP Explainability",
        "Risk Prediction"
    ]

    st.write(" → ".join(workflow))

    st.divider()

    st.subheader("Project Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔍 Risk Prediction")
        st.write("Estimate loan default probability for a new applicant.")

    with col2:
        st.markdown("### 📊 Model Evaluation")
        st.write("Compare models using standard classification metrics.")

    with col3:
        st.markdown("### 🧠 Explainability")
        st.write("Understand which features influence model predictions.")


# ============================================================
# 7. LOAN RISK PREDICTION
# ============================================================

elif page == "Loan Risk Prediction":

    page_header(
        "Loan Risk Prediction",
        "Enter applicant details to estimate the probability of loan default."
    )

    with st.form("prediction_form"):

        st.subheader("Applicant Information")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=28,
                step=1
            )

            income = st.number_input(
                "Annual Income (₹)",
                min_value=1,
                value=600000,
                step=10000
            )

            employment = st.number_input(
                "Employment Length (Years)",
                min_value=0,
                max_value=60,
                value=3,
                step=1
            )

            home = st.selectbox(
                "Home Ownership",
                ["RENT", "OWN", "MORTGAGE", "OTHER"]
            )

            intent = st.selectbox(
                "Loan Purpose",
                [
                    "PERSONAL",
                    "EDUCATION",
                    "MEDICAL",
                    "VENTURE",
                    "HOMEIMPROVEMENT",
                    "DEBTCONSOLIDATION"
                ]
            )

        with col2:

            grade = st.selectbox(
                "Loan Grade",
                ["A", "B", "C", "D", "E", "F", "G"]
            )

            loan_amount = st.number_input(
                "Loan Amount (₹)",
                min_value=1000,
                value=100000,
                step=5000
            )

            interest_rate = st.number_input(
                "Interest Rate (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5
            )

            previous_default = st.selectbox(
                "Previous Default on File",
                ["N", "Y"]
            )

            credit_history = st.number_input(
                "Credit History Length (Years)",
                min_value=0,
                max_value=50,
                value=5,
                step=1
            )

        submitted = st.form_submit_button(
            "Predict Loan Risk",
            width="stretch"
        )

    if submitted:

        applicant = {
            "person_age": age,
            "person_income": income,
            "person_home_ownership": home,
            "person_emp_length": employment,
            "loan_intent": intent,
            "loan_grade": grade,
            "loan_amnt": loan_amount,
            "loan_int_rate": interest_rate,
            "loan_percent_income": loan_amount / income,
            "cb_person_default_on_file": previous_default,
            "cb_person_cred_hist_length": credit_history
        }

        try:

            with st.spinner("Analyzing applicant details..."):

                result = predict_loan_risk(applicant)

            probability = float(result["default_probability"])
            risk = result["risk_level"]
            prediction = int(result["prediction"])

            st.divider()
            st.subheader("Prediction Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Default Probability",
                f"{probability * 100:.2f}%"
            )

            col2.metric(
                "Risk Level",
                str(risk)
            )

            col3.metric(
                "Model Prediction",
                "Default Risk" if prediction == 1 else "No Default Risk"
            )

            st.progress(
                max(0.0, min(1.0, probability))
            )

            if prediction == 1:
                st.error(
                    "The model predicts potential loan default risk."
                )
            else:
                st.success(
                    "The model predicts no loan default."
                )

            st.caption(
                "This is a model-based estimate, not a guaranteed "
                "outcome or a final lending decision."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ============================================================
# 8. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    page_header(
        "Model Performance",
        "Compare machine learning algorithms and evaluate the selected model."
    )

    comparison = load_csv(COMPARISON_PATH)

    if comparison is not None and not comparison.empty:

        st.subheader("Model Comparison")

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        model_column = get_model_column(comparison)

        numeric_columns = comparison.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:

            metric_column = st.selectbox(
                "Choose Evaluation Metric",
                numeric_columns
            )

            chart = px.bar(
                comparison,
                x=model_column,
                y=metric_column,
                color=model_column,
                text_auto=".3f",
                title=f"Model Comparison - {metric_column}"
            )

            chart.update_layout(
                xaxis_title="Machine Learning Model",
                yaxis_title=metric_column,
                showlegend=False
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

    else:
        st.warning("Model comparison results are not available.")

    st.divider()

    st.subheader("Final Test Confusion Matrix")

    if os.path.exists(CONFUSION_MATRIX_PATH):

        st.image(
            CONFUSION_MATRIX_PATH,
            use_container_width=True
        )

    else:
        st.warning("Confusion matrix image was not found.")

    st.caption(
        "The evaluation results describe performance on the saved test set."
    )


# ============================================================
# 9. RISK EXPLAINABILITY
# ============================================================

elif page == "Risk Explainability":

    page_header(
        "Risk Explainability",
        "Understand the features that influence the model's predictions."
    )

    feature_data = load_csv(FEATURE_PATH)

    if feature_data is not None and not feature_data.empty:

        st.subheader("Feature Importance Data")

        st.dataframe(
            feature_data,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("Feature importance data was not found.")

    st.divider()

    if os.path.exists(FEATURE_IMAGE_PATH):

        st.subheader("Feature Importance Visualization")

        st.image(
            FEATURE_IMAGE_PATH,
            use_container_width=True
        )

    else:
        st.warning("Feature importance chart was not found.")

    st.divider()

    if os.path.exists(SHAP_SUMMARY_PATH):

        st.subheader("SHAP Summary Plot")

        st.image(
            SHAP_SUMMARY_PATH,
            use_container_width=True
        )

    else:
        st.warning("SHAP summary plot was not found.")

    st.caption(
        "SHAP explains model influence. Feature importance does not "
        "establish causation."
    )


# ============================================================
# 10. WHAT-IF LOAN SIMULATOR
# ============================================================

elif page == "What-If Loan Simulator":

    page_header(
        "What-If Loan Simulator",
        "Explore how hypothetical changes in applicant details affect "
        "estimated default probability."
    )

    st.info(
        "These are hypothetical model estimates. They are not guaranteed "
        "outcomes or financial advice."
    )

    with st.form("what_if_form"):

        st.subheader("Applicant Information")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=28,
                step=1,
                key="sim_age"
            )

            income = st.number_input(
                "Annual Income (₹)",
                min_value=1,
                value=600000,
                step=10000,
                key="sim_income"
            )

            employment = st.number_input(
                "Employment Length (Years)",
                min_value=0,
                max_value=60,
                value=3,
                step=1,
                key="sim_employment"
            )

            home = st.selectbox(
                "Home Ownership",
                ["RENT", "OWN", "MORTGAGE", "OTHER"],
                key="sim_home"
            )

            intent = st.selectbox(
                "Loan Purpose",
                [
                    "PERSONAL",
                    "EDUCATION",
                    "MEDICAL",
                    "VENTURE",
                    "HOMEIMPROVEMENT",
                    "DEBTCONSOLIDATION"
                ],
                key="sim_intent"
            )

        with col2:

            grade = st.selectbox(
                "Loan Grade",
                ["A", "B", "C", "D", "E", "F", "G"],
                key="sim_grade"
            )

            loan_amount = st.number_input(
                "Loan Amount (₹)",
                min_value=1000,
                value=100000,
                step=5000,
                key="sim_loan"
            )

            interest_rate = st.number_input(
                "Interest Rate (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5,
                key="sim_rate"
            )

            previous_default = st.selectbox(
                "Previous Default on File",
                ["N", "Y"],
                key="sim_default"
            )

            credit_history = st.number_input(
                "Credit History Length (Years)",
                min_value=0,
                max_value=50,
                value=5,
                step=1,
                key="sim_history"
            )

        simulate = st.form_submit_button(
            "Run What-If Simulation",
            use_container_width=True
        )

    if simulate:

        applicant = {
            "person_age": age,
            "person_income": income,
            "person_home_ownership": home,
            "person_emp_length": employment,
            "loan_intent": intent,
            "loan_grade": grade,
            "loan_amnt": loan_amount,
            "loan_int_rate": interest_rate,
            "loan_percent_income": loan_amount / income,
            "cb_person_default_on_file": previous_default,
            "cb_person_cred_hist_length": credit_history
        }

        try:

            with st.spinner("Running simulation scenarios..."):

                results = run_what_if_simulation(applicant)

            if results is None or results.empty:
                st.warning("No simulation results were returned.")

            else:

                st.session_state["what_if_results"] = results

        except Exception as error:
            st.error(f"Simulation failed: {error}")

    # Display saved results after form submission
    if "what_if_results" in st.session_state:

        results = st.session_state["what_if_results"]

        st.divider()
        st.subheader("Scenario Comparison")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        probability_column = get_probability_column(results)
        scenario_column = get_scenario_column(results)

        if probability_column and scenario_column:

            chart = px.bar(
                results,
                x=scenario_column,
                y=probability_column,
                color=probability_column,
                text_auto=".2f",
                title="Default Probability Across Scenarios"
            )

            chart.update_layout(
                xaxis_title="Simulation Scenario",
                yaxis_title="Default Probability (%)",
                xaxis_tickangle=-20
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

        else:
            st.warning(
                "Could not identify the scenario and probability "
                "columns for the chart. The results table is still available."
            )

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Simulation Results",
            data=csv_data,
            file_name="what_if_simulation_results.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.caption(
    "Credit Risk & Loan Default Prediction | "
    "Machine Learning Analytics Dashboard"
)
