# Credit Risk and Loan Default Prediction

## 🚀 Live Demo

Experience the deployed Credit Risk and Loan Default Prediction application:

**[Open Live Application](https://credit-risk-and-loan-default-prediction-kceibex9lyw3xub8appari.streamlit.app/)**

The interactive dashboard allows users to:
- Predict loan default risk.
- Explore model performance.
- Understand predictions using SHAP explainability.
- Simulate hypothetical loan scenarios.

## 1. Project Overview

The **Credit Risk and Loan Default Prediction** project is a machine learning-based application designed to assess the likelihood of a loan applicant defaulting on a loan.

Financial institutions face challenges when evaluating loan applications because approving high-risk applicants can lead to financial losses, while rejecting applicants who could repay their loans may result in missed business opportunities.

This project uses historical loan applicant information and machine learning algorithms to identify potential default risk. It also provides an interactive Streamlit dashboard where users can enter applicant details, obtain risk predictions, explore model performance, understand important prediction factors, and simulate alternative loan scenarios.

The project combines data preprocessing, exploratory data analysis, machine learning, model evaluation, explainable AI, and interactive visualization into one application.

## 2. Problem Statement

Banks and financial institutions need to evaluate the creditworthiness of applicants before approving loans.

Traditional credit assessment processes may involve manual evaluation of income, employment history, loan amount, interest rate, credit history, and other applicant characteristics.

However, evaluating multiple factors manually can be time-consuming and may not consistently identify applicants with a higher probability of default.

The main problem addressed by this project is:

**How can historical loan applicant data be used to predict the probability of loan default and support more informed credit risk assessment?**

### Challenges Addressed

- Identifying applicants who may have a higher probability of default.
- Handling missing values and inconsistent data.
- Understanding relationships between applicant characteristics and loan defaults.
- Managing an imbalanced target variable.
- Comparing multiple machine learning algorithms.
- Evaluating model performance using appropriate classification metrics.
- Making model predictions easier to interpret.
- Understanding how changes in loan application details affect predicted risk.

## 3. Project Objectives

The primary objectives are:

1. Analyze historical loan applicant data.
2. Clean missing, duplicate, and invalid records.
3. Perform exploratory data analysis to identify important patterns.
4. Transform numerical and categorical variables into model-ready features.
5. Train and compare multiple machine learning classification algorithms.
6. Select a model using F1-score as the primary selection metric.
7. Evaluate the selected model using accuracy, precision, recall, F1-score, and ROC-AUC.
8. Explain model predictions using SHAP.
9. Develop a what-if simulator to compare hypothetical loan scenarios.
10. Build an interactive Streamlit dashboard for loan risk assessment.

## 4. Dataset Description

The project uses the Credit Risk Dataset containing historical loan applicant information.

- Original dataset size: 32,581 records.
- Number of original features: 12.
- Target variable: `loan_status`.

### Dataset Features

| Feature | Description |
|---|---|
| person_age | Age of the applicant |
| person_income | Annual income of the applicant |
| person_home_ownership | Applicant's home ownership status |
| person_emp_length | Employment length in years |
| loan_intent | Purpose of the loan |
| loan_grade | Loan grade assigned to the application |
| loan_amnt | Requested loan amount |
| loan_int_rate | Loan interest rate |
| loan_status | Target variable indicating default status |
| loan_percent_income | Loan amount as a proportion of income |
| cb_person_default_on_file | Historical default indicator |
| cb_person_cred_hist_length | Length of credit history |

### Target Variable

- `0` – No Default
- `1` – Default

The target variable is used to train a binary classification model.

## 5. Data Cleaning and Preparation

The original dataset contained missing values, duplicate records, and some invalid values.

The following cleaning operations were performed:

### Missing Value Handling

Missing values represented by `?` were identified in:

- `person_emp_length`
- `loan_int_rate`

### Invalid Value Handling

- Applicant ages above 100 were treated as invalid and marked as missing.
- Employment lengths above 100 years were treated as invalid and marked as missing.
- Other employment-length values were retained.

### Duplicate Removal

A total of 165 exact duplicate records were removed.

### Final Cleaned Dataset

After cleaning:

- Records remaining: 32,416.
- Features remaining: 12.
- Duplicate records remaining: 0.

Missing values after cleaning:

| Feature | Missing Values |
|---|---:|
| person_age | 5 |
| person_emp_length | 889 |
| loan_int_rate | 3,095 |

Missing values were handled during the preprocessing stage using imputation.

## 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the dataset, identify patterns, and examine relationships between applicant characteristics and loan default status.

The following visualizations were generated:

1. Loan Status Distribution
2. Age Distribution
3. Income Distribution
4. Loan Amount Distribution
5. Loan Grade vs Default
6. Loan Intent vs Default
7. Home Ownership vs Default
8. Correlation Heatmap

### Target Distribution

| Loan Status | Records | Percentage |
|---|---:|---:|
| No Default (0) | 25,327 | 78.13% |
| Default (1) | 7,089 | 21.87% |

The dataset contains fewer default cases than non-default cases. This class imbalance was considered when selecting evaluation metrics.

## 7. Data Preprocessing

The dataset was divided into training and testing subsets using stratified sampling.

- Training records: 25,932
- Testing records: 6,484
- Test size: 20%
- Random state: 42

### Numerical Features

The numerical preprocessing pipeline includes:

- Median imputation for missing values.
- Standard scaling using StandardScaler.

### Categorical Features

The categorical preprocessing pipeline includes:

- Most-frequent-value imputation.
- One-hot encoding using OneHotEncoder.
- `handle_unknown="ignore"` to handle previously unseen categories.

A ColumnTransformer was used to combine the numerical and categorical preprocessing pipelines.

The fitted preprocessing object was saved as `preprocessor.pkl` for reuse during predictions.

## 8. Machine Learning Models

Three classification algorithms were trained and evaluated.

### 1. Logistic Regression

Logistic Regression was used as a baseline classification algorithm to estimate the probability of loan default.

### 2. Decision Tree Classifier

Decision Tree was used to identify decision patterns based on applicant and loan characteristics.

### 3. Random Forest Classifier

Random Forest combines multiple decision trees to produce predictions and capture more complex patterns in the dataset.

The Random Forest model was selected based on its F1-score among the evaluated models.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 86.94% | 77.37% | 56.91% | 65.58% | 86.79% |
| Decision Tree | 92.83% | 93.92% | 71.86% | 81.42% | 90.19% |
| Random Forest | 93.40% | 97.60% | 71.58% | 82.59% | 92.80% |

*These are the initial model-comparison results.*

## 9. Final Model Evaluation

The selected Random Forest model was evaluated on the held-out test dataset.

### Final Model Configuration

- Algorithm: Random Forest Classifier
- Number of estimators: 200
- Maximum depth: 15
- Minimum samples split: 5
- Random state: 42
- Primary model selection metric: F1-score

A decision threshold was selected using validation data to optimize F1-score.

### Final Test Results

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 93.28% |
| Precision | 96.76% |
| Recall | 71.65% |
| F1-Score | 82.33% |
| ROC-AUC | 92.53% |

### Confusion Matrix

| | Predicted No Default | Predicted Default |
|---|---:|---:|
| Actual No Default | 5,032 | 34 |
| Actual Default | 402 | 1,016 |

Interpretation:

- True Negatives: 5,032
- False Positives: 34
- False Negatives: 402
- True Positives: 1,016

Precision indicates that most applicants predicted as default cases were actual default cases in the test dataset.

Recall indicates that the model identified approximately 71.65% of the actual default cases.

The model still misses some actual default cases, so predictions should support rather than replace a comprehensive credit assessment.

## 10. Explainable AI Using SHAP

SHAP (SHapley Additive exPlanations) was implemented to understand how different features influence model predictions.

The explainability module generates feature importance results and a SHAP summary visualization.

### Important Features Identified

The leading features by mean absolute SHAP importance include:

| Feature | Mean Absolute SHAP Importance |
|---|---:|
| Loan Percent Income | 0.087573 |
| Person Income | 0.053563 |
| Home Ownership - RENT | 0.042286 |
| Loan Interest Rate | 0.035014 |
| Loan Grade D | 0.031235 |
| Loan Grade A | 0.027944 |
| Loan Grade C | 0.019582 |
| Employment Length | 0.014415 |
| Loan Grade B | 0.013739 |
| Loan Intent - HOMEIMPROVEMENT | 0.013713 |

These values represent model influence, not proof of causation.

SHAP helps make the model more interpretable by showing which features contribute to its predictions.

## 11. What-If Loan Simulator

A what-if simulation module was developed to explore how hypothetical changes in applicant details affect the model's predicted default probability.

The simulator evaluates five scenarios:

1. Original Application
2. Loan Amount Reduced by 20%
3. Interest Rate Reduced by 2%
4. Income Increased by 20%
5. Loan Amount Reduced by 20% and Interest Rate Reduced by 2%

The loan-to-income ratio is recalculated when relevant applicant details change.

Each scenario is passed through the same trained model and preprocessing pipeline.

### Example Simulation Results

| Scenario | Default Probability |
|---|---:|
| Original Application | 6.30% |
| Loan Amount Reduced by 20% | 5.73% |
| Interest Rate Reduced by 2% | 4.37% |
| Income Increased by 20% | 4.84% |
| Loan Amount and Interest Rate Reduced | 3.88% |

These are hypothetical model estimates for the example application. They do not establish that changing a particular loan condition will cause an actual reduction in default risk.

## 12. Streamlit Dashboard

An interactive web application was developed using Streamlit and Plotly.

The dashboard contains five main sections.

### 1. Dashboard Overview

Provides an overview of the credit risk project, model information, and important dataset and model results.

### 2. Loan Risk Prediction

Allows users to enter applicant information and receive:

- Predicted default probability.
- Risk classification.
- Default or no-default prediction.

### 3. Model Performance

Displays model evaluation results, performance metrics, and classification evaluation visualizations.

### 4. Risk Explainability

Displays SHAP-based feature importance and visualizations to help users understand the model.

### 5. What-If Loan Simulator

Allows users to compare different hypothetical loan scenarios and their predicted default probabilities.

## 13. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Preprocessing and machine learning |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualizations |
| Plotly | Interactive charts |
| SHAP | Model explainability |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web dashboard |

## 14. Project Structure

```text
Credit_Risk_Loan_Default_Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── cleaned_credit_risk.csv
│   └── features/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── preprocessor.pkl
│       └── feature_names.json
│
├── models/
│   ├── best_model.pkl
│   ├── decision_tree.pkl
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── results/
│   ├── eda/
│   ├── models/
│   └── explainability/
│
└── src/
    ├── eda.py
    ├── preprocessing.py
    ├── model_training.py
    ├── model_evaluation.py
    ├── explainability.py
    ├── prediction.py
    └── what_if_simulator.py
```

## 15. Installation and Execution

### Prerequisites

- Python 3.10 or later
- Git
- VS Code (recommended)

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Credit_Risk_Loan_Default_Prediction
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will open in your browser, normally at:

```text
http://localhost:8501
```

## 16. Key Outcomes

This project demonstrates an end-to-end machine learning workflow for credit risk prediction.

The key outcomes include:

- Cleaning and preparing 32,416 loan applicant records.
- Performing exploratory data analysis.
- Training and comparing three machine learning algorithms.
- Selecting a Random Forest model based on F1-score.
- Achieving 93.28% test accuracy and 92.53% ROC-AUC.
- Using SHAP to interpret model feature importance.
- Developing a what-if loan simulation module.
- Building an interactive five-section Streamlit dashboard.

## 17. Limitations and Future Enhancements

### Current Limitations

- The model is trained on a particular historical dataset and may not generalize to every financial institution or applicant population.
- Recall is 71.65%, meaning some actual default cases are not detected.
- SHAP feature importance explains model behavior but does not establish causation.
- What-if simulations represent hypothetical model outputs and are not guaranteed financial outcomes.
- The model should not be used as the sole basis for real loan approval or rejection decisions.
- Further validation, fairness analysis, and monitoring are required before real-world financial deployment.

### Future Enhancements

- Hyperparameter tuning and additional model comparisons.
- Cross-validation and improved validation methodology.
- Fairness and bias analysis across applicant groups.
- Model monitoring and performance tracking.
- More detailed applicant-level explanations.
- Integration with secure financial data systems.
- Deployment with production-level security and monitoring.

## 18. Important Disclaimer

This project is developed for educational, portfolio, and machine learning demonstration purposes.

Predictions are generated from historical data and should not be treated as professional financial advice or as the sole basis for lending decisions.

## 19. Author

**M Madhan Mohan Reddy**

Machine Learning | Python | Data Analytics | Streamlit

GitHub: [Add your GitHub Profile URL](https://github.com/)

---

**Project:** Credit Risk and Loan Default Prediction

**Domain:** Banking and Financial Risk Analytics

**Type:** Machine Learning – Binary Classification