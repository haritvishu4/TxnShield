# 🛡️ TxnShield

### Transaction Fraud Intelligence

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Verified-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-33%2F33_Passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TxnShield** is an end-to-end machine learning system for real-time transaction fraud detection and risk intelligence.

It combines fraud probability estimation, explainable risk scoring, SHAP-based model interpretation, FastAPI inference, SQLite audit logging, a Streamlit operations dashboard, automated testing, and Docker containerization.

---

## 📑 Table of Contents

1. [Overview & Problem Statement](#-overview--problem-statement)
2. [Key System Features](#-key-system-features)
3. [End-to-End Architecture](#-end-to-end-architecture)
4. [Dataset & Exploratory Data Analysis](#-dataset--exploratory-data-analysis)
5. [Anti-Leakage Preprocessing](#-anti-leakage-preprocessing)
6. [Machine Learning Models & Imbalance Handling](#-machine-learning-models--imbalance-handling)
7. [Evaluation Metrics & Model Comparison](#-evaluation-metrics--model-comparison)
8. [Decision Threshold Optimization](#-decision-threshold-optimization--generalization)
9. [Dynamic Risk Scoring Engine](#-dynamic-risk-scoring-engine)
10. [Explainable AI with SHAP](#-explainable-ai-with-shap)
11. [FastAPI REST Backend](#-fastapi-rest-backend)
12. [Streamlit Monitoring Dashboard](#-streamlit-monitoring-dashboard)
13. [Database & Audit Trail](#-database--audit-trail)
14. [Installation & Setup](#-installation--setup)
15. [Testing & Verification](#-testing--verification)
16. [Docker Containerization](#-docker-containerization)
17. [Interview Q&A Cheatsheet](#-interview-qa-cheatsheet)
18. [Limitations & Future Roadmap](#-important-limitations--future-roadmap)

---

## 🎯 Overview & Problem Statement

Financial transaction fraud detection is a highly imbalanced machine learning problem.

In the benchmark dataset used by TxnShield, legitimate transactions make up approximately **99.83%** of the data, while fraudulent transactions account for only about **0.17%**.

This creates several important challenges:

1. **Extreme Class Imbalance**  
   A model that predicts every transaction as legitimate can achieve very high accuracy while detecting no fraud at all.

2. **Asymmetric Error Costs**  
   Missing a fraudulent transaction can be significantly more costly than incorrectly flagging a legitimate transaction for additional review.

3. **Explainability Requirements**  
   Risk analysts need more than a probability score. They need interpretable information showing which features contributed to a prediction.

4. **Operational Integration**  
   A practical ML system needs more than a trained model. It also requires APIs, persistence, monitoring, explainability, testing, and deployment support.

TxnShield demonstrates an end-to-end engineering approach to these challenges in a local machine learning fraud-detection prototype.

---

## ✨ Key System Features

- **Multi-Model Benchmarking**  
  Compares Logistic Regression, Random Forest, and XGBoost.

- **Class-Imbalance Handling**  
  Uses class weighting and imbalance-aware training strategies.

- **Leakage-Safe Preprocessing**  
  Scalers are fitted only on the training split before transforming validation and test data.

- **Validation-Tuned Decision Threshold**  
  Uses a threshold of:

  ```text
  0.4159