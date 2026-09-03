import json

def make_eda_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🛡️ Real-Time Fraud Detection & Risk Intelligence\n",
                    "## Notebook 01: Exploratory Data Analysis (EDA) & Imbalance Analysis\n",
                    "\n",
                    "This notebook provides exploratory analysis on the credit card transactions dataset, detailing:\n",
                    "- Severe class imbalance (~0.17% fraud rate)\n",
                    "- Amount and Time feature distributions\n",
                    "- PCA feature correlations with transaction fraud\n",
                    "- Strategies for leakage prevention and metric selection"
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
                    "from src.data.ingestion import DataIngestion\n",
                    "\n",
                    "sns.set_theme(style='whitegrid', palette='muted')\n",
                    "%matplotlib inline"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 1. Load Dataset and Inspect Schema"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "ingestion = DataIngestion()\n",
                    "df = ingestion.load_data()\n",
                    "print(f'Total Rows: {df.shape[0]:,}, Total Columns: {df.shape[1]}')\n",
                    "display(df.head())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 2. Class Imbalance: Legit (0) vs Fraud (1)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "counts = df['Class'].value_counts()\n",
                    "percentages = df['Class'].value_counts(normalize=True) * 100\n",
                    "print(f'Legitimate (Class 0): {counts[0]:,} ({percentages[0]:.3f}%)')\n",
                    "print(f'Fraudulent (Class 1): {counts[1]:,} ({percentages[1]:.3f}%)')\n",
                    "print(f'Imbalance Ratio: 1 fraud for every {counts[0]//counts[1]:,} legitimate transactions')\n",
                    "\n",
                    "fig, ax = plt.subplots(figsize=(6, 4))\n",
                    "sns.countplot(x='Class', data=df, ax=ax, palette=['#43a047', '#e53935'])\n",
                    "ax.set_yscale('log')\n",
                    "ax.set_title('Transaction Distribution (Log Scale)')\n",
                    "ax.set_xticklabels(['Legitimate (0)', 'Fraud (1)'])\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 3. Transaction Amount Distribution"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Amount Summary for Legitimate Transactions:')\n",
                    "print(df[df['Class'] == 0]['Amount'].describe())\n",
                    "\n",
                    "print('\\nAmount Summary for Fraudulent Transactions:')\n",
                    "print(df[df['Class'] == 1]['Amount'].describe())\n",
                    "\n",
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
                    "sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, kde=True, ax=ax1, color='#43a047')\n",
                    "ax1.set_title('Legit Amount Distribution')\n",
                    "ax1.set_xlim(0, 1000)\n",
                    "\n",
                    "sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, kde=True, ax=ax2, color='#e53935')\n",
                    "ax2.set_title('Fraud Amount Distribution')\n",
                    "ax2.set_xlim(0, 1000)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 4. Correlation with Target Class"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "corr = df.corr()['Class'].drop('Class').sort_values()\n",
                    "print('Top 5 Inversely Correlated Features (Lower value -> Higher Fraud Risk):')\n",
                    "print(corr.head(5))\n",
                    "print('\\nTop 5 Positively Correlated Features (Higher value -> Higher Fraud Risk):')\n",
                    "print(corr.tail(5))\n",
                    "\n",
                    "fig, ax = plt.subplots(figsize=(10, 6))\n",
                    "corr.plot(kind='bar', ax=ax, color=np.where(corr>0, '#e53935', '#1e88e5'))\n",
                    "ax.set_title('Correlation of PCA & Engineered Features with Target Class')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open("notebooks/01_exploratory_data_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

def make_modeling_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🛡️ Real-Time Fraud Detection & Risk Intelligence\n",
                    "## Notebook 02: Model Benchmarking, Threshold Optimization & SHAP Explainability\n",
                    "\n",
                    "This notebook demonstrates:\n",
                    "1. Stratified Train / Validation / Test data splitting preventing leakage\n",
                    "2. Benchmarking Logistic Regression, Random Forest, and XGBoost\n",
                    "3. Evaluating Precision-Recall Curves (PR-AUC) and ROC-AUC\n",
                    "4. Optimal Threshold Search on validation split to maximize F1\n",
                    "5. SHAP feature attribution waterfall plots"
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
                    "import shap\n",
                    "from src.data.ingestion import DataIngestion\n",
                    "from src.data.preprocessor import DataPreprocessor\n",
                    "from src.models.trainer import ModelTrainer\n",
                    "from src.models.evaluator import ModelEvaluator\n",
                    "from src.models.threshold_optimizer import ThresholdOptimizer\n",
                    "from src.models.explainability import FraudExplainer\n",
                    "%matplotlib inline"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 1. Leakage-Free Preprocessing"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "ingestion = DataIngestion()\n",
                    "df = ingestion.load_data()\n",
                    "preprocessor = DataPreprocessor()\n",
                    "X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = preprocessor.split_data(df)\n",
                    "X_train, X_val, X_test = preprocessor.fit_transform(X_train_df, X_val_df, X_test_df)\n",
                    "print(f'Transformed Train Shape: {X_train.shape}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 2. Multi-Model Benchmark"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "trainer = ModelTrainer()\n",
                    "lr = trainer.train_logistic_regression(X_train, y_train.values)\n",
                    "rf = trainer.train_random_forest(X_train, y_train.values)\n",
                    "xgb = trainer.train_xgboost(X_train, y_train.values)\n",
                    "\n",
                    "models = {'Logistic Regression': lr, 'Random Forest': rf, 'XGBoost': xgb}\n",
                    "val_metrics = {name: ModelEvaluator.evaluate(m, X_val, y_val.values) for name, m in models.items()}\n",
                    "display(ModelEvaluator.compare_models(val_metrics))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 3. Threshold Optimization for Best Model"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "best_model = xgb\n",
                    "opt_result = ThresholdOptimizer.find_optimal_threshold(best_model, X_val, y_val.values)\n",
                    "print(f'Optimal Threshold: {opt_result[\"optimal_threshold\"]}')\n",
                    "print(f'Optimal F1-Score: {opt_result[\"best_f_score\"]}')\n",
                    "\n",
                    "df_curve = pd.DataFrame(opt_result['curve_data'])\n",
                    "plt.figure(figsize=(8, 4))\n",
                    "plt.plot(df_curve['threshold'], df_curve['precision'], label='Precision')\n",
                    "plt.plot(df_curve['threshold'], df_curve['recall'], label='Recall')\n",
                    "plt.plot(df_curve['threshold'], df_curve['f_beta'], label='F1-Score', linewidth=2)\n",
                    "plt.axvline(opt_result['optimal_threshold'], color='r', linestyle='--', label=f'Optimal τ* = {opt_result[\"optimal_threshold\"]}')\n",
                    "plt.title('Threshold Optimization Trade-off Curve')\n",
                    "plt.xlabel('Decision Threshold')\n",
                    "plt.ylabel('Score')\n",
                    "plt.legend()\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open("notebooks/02_model_experimentation.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

if __name__ == "__main__":
    make_eda_notebook()
    make_modeling_notebook()
    print("Notebooks successfully generated.")
