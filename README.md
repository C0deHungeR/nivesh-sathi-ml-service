---

# 📊 Mutual Fund Recommendation ML Service

## **NiveshSathi – AI-Driven Mutual Fund Recommendation Engine**

---

## 📌 Overview

This repository contains the **Machine Learning (ML) service** powering the **Mutual Fund Recommendation** feature of the **NiveshSathi** platform.

The system provides **realistic, risk-aware, and explainable mutual fund recommendations** based on investor inputs such as:

* AMC name
* Category
* Investment mode (Lump Sum or SIP)
* Investment amount
* Investment tenure

The ML service is exposed via a **FastAPI REST API** and is designed to be consumed by a **Spring Boot backend / Web frontend**.

---

## 🎯 Problem Statement

Retail investors often struggle to:

* Select appropriate mutual fund schemes
* Understand expected future performance
* Evaluate risk and uncertainty realistically

This ML service addresses the problem by:

* Filtering eligible schemes using **rule-based constraints**
* Forecasting future **1-year returns using ML**
* Modeling uncertainty using **Monte Carlo simulation**
* Returning **Top-5 robust recommendations** instead of a single suggestion

---

## 🧠 Solution Architecture

This is a **Hybrid Recommendation System**, combining deterministic rules with machine learning.

```
Investor Input
   ↓
Rule-Based Filtering (Eligibility)
   ↓
XGBoost Regression (Expected Return)
   ↓
Monte Carlo Simulation (Uncertainty)
   ↓
Ranking & Scoring
   ↓
Top 5 Recommendations (API Response)
```

---

## 🧩 Key Components

### 1️⃣ Rule-Based Filtering

Ensures:

* AMC and category match
* Investment amount meets `min_lumpsum` or `min_sip`
* SIP and Lump Sum are mutually exclusive
* No negative-growth schemes are recommended
* Funds are mature and cost-efficient

This step guarantees **business correctness and explainability**.

---

### 2️⃣ Machine Learning Model 

* **Model Type:** Deep Learning Sequential Model
* **Target:** Change (delta) in 1-year return

#### Why Delta Prediction?

* Avoids copying past returns
* Produces conservative and realistic forecasts
* Reduces overfitting

#### Input Features

* `amc_name`
* `category`
* `fund_age_yr`
* `expense_ratio`
* `sd`
* `alpha`
* `sharpe`
* `tenure`

#### Saved Artifacts

```
xgb_model.pkl
preprocessor.pkl
```

---

### 3️⃣ Monte Carlo Simulation

Monte Carlo simulation is applied at **inference time**, not during training.

#### Purpose

* Model uncertainty of future returns
* Generate a realistic range instead of a single value

#### Important Design Decisions

* Simulation models **1-year returns only**
* Investment tenure adjusts **volatility**, not returns
* Hard caps prevent unrealistic outputs

#### Outputs

* Expected return
* Best-case scenario (90th percentile)
* Worst-case scenario (10th percentile)

---

### 4️⃣ Ranking & Top-5 Recommendation

Each eligible scheme is scored using:

```
Final Score =
0.5 × Expected Return
+ 0.3 × Past 1-Year Return
+ 0.2 × Sharpe Ratio
```

This ensures:

* Stability
* Risk-adjusted performance
* Avoidance of overly aggressive funds

Only the **Top 5 schemes** are returned.

---

## 📂 Dataset

**Source:** Mutual Funds India (Hackathon Dataset)

### Key Columns Used

* `scheme_name`
* `amc_name`
* `category`
* `min_lumpsum`
* `min_sip`
* `fund_age_yr`
* `expense_ratio`
* `sd`
* `sharpe`
* `alpha`
* `risk_level`
* `returns_1yr`

### Preprocessing Steps

* Replaced `-` with `NaN`
* Converted numeric columns safely
* Removed schemes with negative 1-year returns
* Removed incomplete or invalid records

---

## 🔐 Input Validation Rules

* Exactly **one** must be provided:

  * Lump sum investment
  * SIP investment
* Tenure must be between **1 and 50 years**
* Invalid requests are rejected using **Pydantic v2 validators**

---

## 🔌 API Specification

### Endpoint

```
POST /predict
```

### Request Body (Lump Sum)

```json
{
  "amc_name": "Bank of India Mutual Fund",
  "category": "Debt",
  "investment_amount": 5000000,
  "tenure": 10
}
```

### Request Body (SIP)

```json
{
  "amc_name": "HDFC Mutual Fund",
  "category": "Equity",
  "sip_amount": 20000,
  "tenure": 15
}
```

### Response

```json
{
  "recommendations": [
    {
      "scheme_name": "Bank of India Conservative Hybrid Fund",
      "past_1yr": 12.4,
      "expected": 9.8,
      "best_case": 15.6,
      "worst_case": 4.9,
      "risk": "Medium"
    }
  ]
}
```

---

## 🚀 How to Run the ML Service

### Install Dependencies

```bash
pip install fastapi uvicorn pandas numpy scikit-learn xgboost
```

### Required Files

```
MF_India_AI.csv
xgb_model.pkl
preprocessor.pkl
main.py
```

### Start the Service

```bash
uvicorn main:app --reload --port 8000
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Testing on Google Colab

* Upload required files
* Run `main.py`
* Call the `predict()` function with sample inputs

---

## ⚖️ Design Assumptions & Limitations

* Market returns are uncertain
* Forecasts are conservative
* Results are **not financial advice**
* Focus on realistic **1-year projections**
* Long-term tenure affects confidence, not compounding

---

## 🏆 Why This Approach Is Credible

* Separates rules from ML
* Uses ML only where it adds value
* Monte Carlo used for uncertainty, not hype
* Avoids unrealistic returns
* Fully explainable and auditable

---

## 📈 Future Improvements

* Personalized risk profiling
* Multi-year CAGR modeling
* SIP cash-flow simulation
* SHAP-based explainability
* Automated model retraining

---
