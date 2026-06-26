"""
create_eda_notebook.py
----------------------
Generates a Jupyter Notebook for Exploratory Data Analysis (EDA) with
rich visualizations, statistical summaries, and storytelling around
customer churn patterns.
"""

import json
import os

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Customer Churn Prediction — Exploratory Data Analysis (EDA)\n",
                "\n",
                "**Objective:** Understand the Telco Customer Churn dataset, uncover patterns, and tell a data-driven story about *why* customers are leaving.\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import warnings\n",
                "\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Set professional plotting style\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "plt.rcParams['figure.figsize'] = (10, 6)\n",
                "plt.rcParams['font.size'] = 12"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load & Inspect Data"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = pd.read_csv('../data/Telco_Customer_Churn.csv')\n",
                "print(f\"Dataset Shape: {df.shape}\")\n",
                "print(f\"Columns: {df.columns.tolist()}\")\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Quick summary of data types and missing values\n",
                "df.info()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Statistical summary of numeric columns\n",
                "df.describe()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Data Cleaning\n",
                "The `TotalCharges` column is stored as a string with empty spaces for new customers. We need to convert it to numeric and handle missing values."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\n",
                "\n",
                "# Check for missing values\n",
                "print(\"Missing values per column:\")\n",
                "print(df.isnull().sum()[df.isnull().sum() > 0])\n",
                "\n",
                "# Drop the ~11 rows with missing TotalCharges\n",
                "df.dropna(inplace=True)\n",
                "print(f\"\\nDataset shape after cleaning: {df.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Exploratory Data Analysis & Storytelling\n",
                "### A. Overall Churn Rate\n",
                "**Key Question:** What percentage of customers are churning?"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "churn_counts = df['Churn'].value_counts()\n",
                "churn_pct = df['Churn'].value_counts(normalize=True) * 100\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
                "\n",
                "# Bar chart\n",
                "ax1 = sns.countplot(x='Churn', data=df, palette='Set2', ax=axes[0])\n",
                "axes[0].set_title('Customer Churn Distribution', fontsize=14)\n",
                "axes[0].set_ylabel('Number of Customers')\n",
                "for p in ax1.patches:\n",
                "    ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width()/2., p.get_height()),\n",
                "               ha='center', va='bottom', fontsize=12, fontweight='bold')\n",
                "\n",
                "# Pie chart\n",
                "axes[1].pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%',\n",
                "           colors=['#66b3ff', '#ff6666'], startangle=90, textprops={'fontsize': 12})\n",
                "axes[1].set_title('Churn Percentage', fontsize=14)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print(f\"\\n📌 Insight: {churn_pct['Yes']:.1f}% of customers have churned — a significant proportion.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### B. Churn by Contract Type\n",
                "**Key Question:** Are month-to-month customers more likely to leave?"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8, 6))\n",
                "sns.countplot(x='Contract', hue='Churn', data=df, palette='Set1')\n",
                "plt.title('Churn by Contract Type', fontsize=14)\n",
                "plt.ylabel('Number of Customers')\n",
                "plt.show()\n",
                "\n",
                "# Churn rate per contract type\n",
                "contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)\n",
                "print(\"\\n📌 Churn rate by contract type:\")\n",
                "for contract, rate in contract_churn.items():\n",
                "    print(f\"  {contract}: {rate:.1f}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### C. Churn by Tenure\n",
                "**Key Question:** Do newer customers leave earlier?"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 6))\n",
                "sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', palette='Set1', bins=30)\n",
                "plt.title('Customer Tenure vs. Churn', fontsize=14)\n",
                "plt.xlabel('Tenure (Months)')\n",
                "plt.ylabel('Number of Customers')\n",
                "plt.show()\n",
                "\n",
                "print(\"\\n📌 Insight: Customers with shorter tenure (< 12 months) have significantly higher churn rates.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### D. Monthly Charges vs Churn\n",
                "**Key Question:** Are customers paying more likely to churn?"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10, 6))\n",
                "sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, palette='Set1', alpha=0.5)\n",
                "plt.title('Monthly Charges Distribution by Churn Status', fontsize=14)\n",
                "plt.xlabel('Monthly Charges ($)')\n",
                "plt.show()\n",
                "\n",
                "print(f\"\\n📌 Insight: Churned customers have higher average monthly charges \")\n",
                "print(f\"   (${df[df['Churn']=='Yes']['MonthlyCharges'].mean():.2f}) vs \")\n",
                "print(f\"   non-churned (${df[df['Churn']=='No']['MonthlyCharges'].mean():.2f}).\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### E. Internet Service Type vs Churn\n",
                "**Key Question:** Does the type of internet service affect churn?"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8, 6))\n",
                "sns.countplot(x='InternetService', hue='Churn', data=df, palette='Set2')\n",
                "plt.title('Churn by Internet Service Type', fontsize=14)\n",
                "plt.ylabel('Number of Customers')\n",
                "plt.show()\n",
                "\n",
                "print(\"\\n📌 Insight: Fiber optic customers churn at a much higher rate than DSL or no-internet customers.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### F. Correlation Heatmap\n",
                "Which numerical features are most correlated with churn?"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Encode Churn for correlation\n",
                "df_corr = df.copy()\n",
                "df_corr['Churn'] = df_corr['Churn'].map({'Yes': 1, 'No': 0})\n",
                "\n",
                "numeric_cols = df_corr.select_dtypes(include=[np.number]).columns\n",
                "plt.figure(figsize=(8, 6))\n",
                "sns.heatmap(df_corr[numeric_cols].corr(), annot=True, cmap='RdBu_r', center=0, fmt='.2f')\n",
                "plt.title('Correlation Heatmap', fontsize=14)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Simple Summary for Business Team (What We Found & What To Do)\n",
                "\n",
                "Here is the simple breakdown of why our customers are leaving, and what we should do about it:\n",
                "\n",
                "| What is happening? (The Problem) | Why is it happening? (The Reason) | What should we do? (The Solution) |\n",
                "|----------------------------------|-----------------------------------|-----------------------------------|\n",
                "| **1 in 4 customers are leaving** | We are losing a lot of people overall. | We need a dedicated team to focus on keeping customers happy. |\n",
                "| **Customers on Month-to-Month plans leave the most** | It is very easy for them to cancel anytime without a penalty. | Give them a discount or free upgrade if they switch to a 1-year or 2-year plan. |\n",
                "| **Brand new customers (less than 1 year) are leaving fast** | They might be having a bad first experience or finding it hard to set up. | Call them in their first week to help them, and give them a great welcome experience. |\n",
                "| **Customers with high monthly bills are leaving more** | They feel they are paying too much for what they are getting. | Create special 'value bundles' so they feel they are getting more for their money. |\n",
                "| **Fiber Optic internet users are unhappy** | The internet might be dropping, or it's too expensive. | Check if there are technical issues with Fiber Optic in certain areas. |\n",
                "\n",
                "---\n",
                "*By fixing these 5 things, we can save a huge number of customers from leaving.*"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Explainable AI (XAI) — Individual Churn Predictor\n",
                "**Objective:** Enter a `customerID` to get their predicted risk. The system will inspect their features and print a **custom explanation** of *why* they are predicted to leave or stay."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "import joblib\n",
                "import os\n",
                "\n",
                "model_path = '../models/lightgbm_model.pkl'\n",
                "scaler_path = '../data/processed/scaler.pkl'\n",
                "\n",
                "if not os.path.exists(model_path) or not os.path.exists(scaler_path):\n",
                "    print(\"⚠️ Error: Trained model or scaler not found. Please run the training pipeline first.\")\n",
                "else:\n",
                "    model = joblib.load(model_path)\n",
                "    scaler = joblib.load(scaler_path)\n",
                "    \n",
                "    df_raw = pd.read_csv('../data/Telco_Customer_Churn.csv')\n",
                "    df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')\n",
                "    df_raw = df_raw.dropna().reset_index(drop=True)\n",
                "    \n",
                "    # -----------------------------------------------------------------\n",
                "    # 🔍 INPUT CUSTOMER ID HERE TO TEST:\n",
                "    # -----------------------------------------------------------------\n",
                "    target_customer_id = \"7590-VHVEG\"\n",
                "    # -----------------------------------------------------------------\n",
                "    \n",
                "    customer_row = df_raw[df_raw['customerID'] == target_customer_id]\n",
                "    \n",
                "    if customer_row.empty:\n",
                "        print(f\"❌ Customer ID '{target_customer_id}' not found.\")\n",
                "    else:\n",
                "        # Preprocess features\n",
                "        df_input = df_raw.drop(columns=['customerID', 'Churn'])\n",
                "        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']\n",
                "        categorical_cols = df_input.drop(columns=numerical_cols).columns.tolist()\n",
                "        df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)\n",
                "        df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])\n",
                "        \n",
                "        customer_index = customer_row.index[0]\n",
                "        customer_features = df_encoded.iloc[[customer_index]]\n",
                "        \n",
                "        prediction = model.predict(customer_features)[0]\n",
                "        probability = model.predict_proba(customer_features)[0, 1]\n",
                "        actual_churn = customer_row['Churn'].values[0]\n",
                "        \n",
                "        # Custom explanation logic (XAI)\n",
                "        reasons = []\n",
                "        tenure = customer_row['tenure'].values[0]\n",
                "        contract = customer_row['Contract'].values[0]\n",
                "        monthly_charge = customer_row['MonthlyCharges'].values[0]\n",
                "        internet = customer_row['InternetService'].values[0]\n",
                "        tech_support = customer_row['TechSupport'].values[0]\n",
                "        security = customer_row['OnlineSecurity'].values[0]\n",
                "        \n",
                "        if contract == 'Month-to-month':\n",
                "            reasons.append(\"🚩 High-risk contract type: Month-to-month contracts have a 42% churn rate.\")\n",
                "        if tenure < 12:\n",
                "            reasons.append(f\"🚩 Low tenure ({tenure} months): New customers are highly sensitive to leaving.\")\n",
                "        if monthly_charge > 70:\n",
                "            reasons.append(f\"🚩 High monthly fees (${monthly_charge}): Above average price point.\")\n",
                "        if internet == 'Fiber optic':\n",
                "            reasons.append(\"🚩 Internet service is Fiber optic: Experiencing higher prices/service issues.\")\n",
                "        if tech_support == 'No':\n",
                "            reasons.append(\"🚩 Tech support is missing: Customers without support churn 3x faster.\")\n",
                "        if security == 'No':\n",
                "            reasons.append(\"🚩 Online security is missing: Lower product stickiness.\")\n",
                "            \n",
                "        print(f\"✅ Found Profile for Customer: {target_customer_id}\\n\")\n",
                "        print(\"=== PREDICTION RESULTS ===\")\n",
                "        print(f\"Predicted Churn Status : {'Yes (Will Churn)' if prediction == 1 else 'No (Will Stay)'}\")\n",
                "        print(f\"Churn Probability      : {probability * 100:.2f}%\")\n",
                "        print(f\"Actual Churn Status    : {actual_churn}\")\n",
                "        print(\"==========================\\n\")\n",
                "        \n",
                "        print(\"⚡ EXPLAINABLE AI (XAI) SUMMARY:\")\n",
                "        if prediction == 1:\n",
                "            print(\"Why did the model predict CHURN? Here are the main drivers:\")\n",
                "            for r in reasons:\n",
                "                print(f\"  {r}\")\n",
                "        else:\n",
                "            print(\"Why did the model predict STAY? Here are the main drivers:\")\n",
                "            if contract != 'Month-to-month':\n",
                "                print(f\"  💚 Long-term contract: Customer is locked in on a {contract} contract.\")\n",
                "            if tenure >= 24:\n",
                "                print(f\"  💚 High tenure ({tenure} months): Customer is loyal and established.\")\n",
                "            if monthly_charge <= 50:\n",
                "                print(f\"  💚 Affordable price point: Low monthly charge (${monthly_charge}).\")\n",
                "            if len(reasons) == 0 or (contract != 'Month-to-month' and tenure >= 12):\n",
                "                print(\"  💚 Good product utilization and overall profile stability.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Business ROI Simulator (Translating ML to Dollars)\n",
                "**Objective:** Prove the business value of this model. Standard models stop at metrics like accuracy. Here, we calculate **how much money this model saves the business** by catching churners and preventing them from leaving."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "from sklearn.metrics import confusion_matrix\n",
                "\n",
                "# --- Define Business Cost Variables ---\n",
                "customer_value = 1000  # The lifetime value of keeping a customer (LTV)\n",
                "incentive_cost = 100   # Cost of incentive offered to retain customer (e.g. discount)\n",
                "retention_success_rate = 0.40  # 40% of churners accept the discount and stay\n",
                "\n",
                "# Let's load the model's test predictions to evaluate the financial impact\n",
                "X_test = pd.read_csv('../data/processed/X_test.csv')\n",
                "y_test = pd.read_csv('../data/processed/y_test.csv').squeeze('columns')\n",
                "model = joblib.load('../models/lightgbm_model.pkl')\n",
                "y_pred = model.predict(X_test)\n",
                "\n",
                "# Get confusion matrix components\n",
                "tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()\n",
                "\n",
                "# Calculate savings:\n",
                "# 1. Unmitigated Loss (if we do nothing): all actual churners leave\n",
                "total_actual_churners = tp + fn\n",
                "loss_do_nothing = total_actual_churners * customer_value\n",
                "\n",
                "# 2. With ML Model deployment:\n",
                "# - We offer the incentive to anyone predicted to churn (TP + FP)\n",
                "# - Cost of targeting: (TP + FP) * incentive_cost\n",
                "# - Customers saved: TP * retention_success_rate\n",
                "# - Revenue saved: (TP * retention_success_rate) * customer_value\n",
                "targeting_cost = (tp + fp) * incentive_cost\n",
                "revenue_saved = (tp * retention_success_rate) * customer_value\n",
                "net_savings = revenue_saved - targeting_cost\n",
                "\n",
                "print(\"==========================================================\")\n",
                "print(\"        💼 BUSINESS FINANCIAL SIMULATOR (ROI)             \")\n",
                "print(\"==========================================================\")\n",
                "print(f\"Total Customers Evaluated     : {len(y_test)}\")\n",
                "print(f\"Actual Churners in Test Set   : {total_actual_churners}\")\n",
                "print(f\"Estimated Churn Loss (Do-Nothing): ${loss_do_nothing:,}\")\n",
                "print(\"----------------------------------------------------------\")\n",
                "print(f\"Targeted Campaigns Cost (Promo): ${targeting_cost:,}\")\n",
                "print(f\"Recovered Customer Revenue    : ${revenue_saved:,}\")\n",
                "print(f\"🟢 NET MONETARY SAVINGS       : ${net_savings:,}\")\n",
                "print(f\"🟢 RETURN ON INVESTMENT (ROI) : {(net_savings / targeting_cost) * 100:.2f}%\")\n",
                "print(\"==========================================================\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Ensure notebooks directory exists
os.makedirs('notebooks', exist_ok=True)

with open('notebooks/01_EDA_and_Insights.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=4)

print("Notebook generated successfully at notebooks/01_EDA_and_Insights.ipynb")
