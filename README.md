# 📊 Telecom Customer Churn Prediction

> **Predict which telecom customers are likely to leave — and understand *why* — using Machine Learning.**

---

## 📖 Project Description
This is an end-to-end Machine Learning project designed to predict customer churn (subscription cancellation) for a telecom provider. By analyzing historical customer profiles, usage habits, and billing details, the system classifies whether a customer will leave or stay, and calculates a percentage probability of their churn risk. This enables retention teams to target high-risk customers proactively with customized loyalty offers.

---

## 💡 Solution Approach
Our solution follows a modular data science pipeline:
1. **Data Preprocessing (`data_preprocessing.py`):**
   * Converts `TotalCharges` from string to numeric and drops sparse missing rows (~11 rows).
   * Encodes categorical columns (e.g. Contract, Payment Method) using One-Hot Encoding (`drop_first=True` to prevent multicollinearity).
   * Splitting data (80/20 train-test ratio, stratified to balance the 26.5% churn distribution).
   * Scales numerical values (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` to ensure unbiased model weights.
2. **Model Training (`model_training.py`):**
   * Trains **Logistic Regression** to establish a simple, fast baseline.
   * Trains **LightGBM** (Gradient Boosting Classifier) to capture complex, non-linear relationships.
3. **Evaluation (`evaluation.py`):**
   * Computes classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
   * Generates confusion matrices, ROC curves, and a **Feature Importance** plot showing key churn drivers.
4. **Visual Analytics & Predictor (`01_EDA_and_Insights.ipynb`):**
   * Interactive storytelling dashboard containing charts on churn distribution, monthly charges, contract types, and internet services.
   * Features a **real-time interactive lookup tool** where users can input any Customer ID to retrieve their profile and run predictions instantly.

---

## 🛠️ Prerequisites & Dependencies
* **Python version:** Python 3.8 or higher.
* **Dependencies:** Listed in [`requirements.txt`](file:///c:/Users/SARANKUMAR/Downloads/Customer%20Churn%20Prediction/requirements.txt):
  * `pandas` & `numpy` (Data manipulation)
  * `scikit-learn` (Preprocessing & metrics)
  * `lightgbm` (Machine Learning model)
  * `matplotlib` & `seaborn` (Data visualizations)
  * `joblib` (Model serialization)
  * `jupyter` (Interactive notebook interface)

---

## 💻 Setup & Usage Instructions

### 1. Environment Setup
Clone the repository and run the following in your terminal:
```powershell
# Create a virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1   # On Windows (PowerShell)
# venv\Scripts\activate.bat   # On Windows (CMD)
# source venv/bin/activate    # On macOS/Linux

# Install required dependencies
pip install -r requirements.txt
```

### 2. Usage — Running the Project

#### Run the Entire Pipeline
Run the following runner script to preprocess data, train models, evaluate outputs, and generate the notebook:
```powershell
python run_pipeline.py
```

#### Run the Interactive & Business Simulation Tools
To explore the visual insights, test predictions, and view ROI estimates:
```powershell
jupyter notebook notebooks/01_EDA_and_Insights.ipynb
```
* Once open in your browser, select **Cell** $\rightarrow$ **Run All**.
* Scroll to the bottom to interact with the standout features:
  * **Section 5 (Explainable AI Predictor):** Change `target_customer_id = "7590-VHVEG"` to see a live predicted churn probability along with a custom text explanation of *why* the model predicted Churn or Stay.
  * **Section 6 (Business ROI Simulator):** View the live financial impact simulation showing how many dollars/rupees the ML model saves the company compared to doing nothing.

---

## 📈 Model Performance & Results
* **Accuracy:** **80%** (overall correct predictions).
* **ROC-AUC:** **0.83+** (excellent capability to distinguish churners from loyal customers).
* **Key Drivers Identified:** Month-to-month contracts, short tenure (<12 months), and lack of online security/tech support are the primary indicators of churn.

---

## 🌟 Standout Features (What Makes This Project Special)
1. **Explainable AI (XAI):** Instead of a "black box" prediction, the model outputs specific reasons why a customer is flagged (e.g. *"Tech support is missing: Customers without support churn 3x faster"*).
2. **Business ROI Simulator:** Bridges the gap between ML metrics and business finance. It translates test metrics into **net monetary savings** based on customer lifetime value (LTV) and retention campaign costs. 

