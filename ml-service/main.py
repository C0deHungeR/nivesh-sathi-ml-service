from fastapi import FastAPI
from pydantic import BaseModel, model_validator
import pandas as pd
import numpy as np
import pickle

# ======================================================
# CONFIG (REALISTIC FINANCIAL BOUNDS)
# ======================================================
MIN_RETURN = 4      # minimum reasonable 1Y MF return (%)
MAX_RETURN = 25     # maximum reasonable expected 1Y return (%)
MAX_MC_RETURN = 40  # absolute cap for Monte Carlo

# ======================================================
# CATEGORY MAPPING (UI → DATASET)
# ======================================================
UI_TO_DATASET_CATEGORY = {
    "Large Cap": "Equity",
    "Mid Cap": "Equity",
    "Small Cap": "Equity",
    "Hybrid": "Hybrid",
    "Debt": "Debt",
    "All Categories": None
}
class TFWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)


# ======================================================
# LOAD MODEL & PREPROCESSOR
# ======================================================
model = pickle.load(open("xgb_model.pkl", "rb"))          # PyTorch or TF wrapped model
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

# ======================================================
# LOAD & CLEAN DATASET
# ======================================================
df = pd.read_csv("MF_India_AI.csv")
df.replace("-", np.nan, inplace=True)

numeric_cols = [
    "min_sip", "min_lumpsum", "expense_ratio", "fund_size_cr",
    "fund_age_yr", "sortino", "alpha", "sd", "beta",
    "sharpe", "returns_1yr"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(
    subset=["scheme_name", "amc_name", "category", "returns_1yr"],
    inplace=True
)

# ❗ Never recommend negative growth schemes
df = df[df["returns_1yr"] > 0]

BASE_RETURN = df["returns_1yr"].median()

# ======================================================
# FASTAPI APP
# ======================================================
app = FastAPI(title="NiveshSathi ML Service (Production Safe)")

# ======================================================
# INPUT SCHEMA (Pydantic v2 SAFE)
# ======================================================
class InvestorInput(BaseModel):
    amc_name: str
    category: str
    investment_amount: float | None = None   # Lumpsum
    sip_amount: float | None = None          # SIP
    tenure: int                              # years (<=50)

    @model_validator(mode="after")
    def validate_investment_mode(self):
        if self.investment_amount and self.sip_amount:
            raise ValueError("Choose either lump sum OR SIP, not both.")
        if not self.investment_amount and not self.sip_amount:
            raise ValueError("Either lump sum or SIP must be provided.")
        if self.tenure <= 0 or self.tenure > 50:
            raise ValueError("Tenure must be between 1 and 50 years.")
        return self

# ======================================================
# RISK MAPPING
# ======================================================
def map_risk_level(rank: int) -> str:
    if rank <= 2:
        return "Low"
    elif rank <= 4:
        return "Medium"
    else:
        return "High"

# ======================================================
# MONTE CARLO (1-YEAR, REALISTIC)
# ======================================================
def monte_carlo(mean_return, volatility, tenure, n=1000):
    mean_return = np.clip(mean_return, MIN_RETURN, MAX_RETURN)

    # Long-term investors experience smoother volatility
    tenure_factor = max(0.5, min(1.0, 5 / tenure))
    adjusted_vol = volatility * tenure_factor

    sims = np.random.normal(mean_return, adjusted_vol, n)
    sims = np.clip(sims, 0, MAX_MC_RETURN)

    return {
        "expected": round(float(np.mean(sims)), 2),
        "best_case": round(float(np.percentile(sims, 90)), 2),
        "worst_case": round(float(np.percentile(sims, 10)), 2)
    }

# ======================================================
# PREDICTION ENDPOINT
# ======================================================
@app.post("/predict")
def predict(data: InvestorInput):

    mapped_category = UI_TO_DATASET_CATEGORY.get(data.category)

    # --------------------------------------------------
    # FILTERING (AMC + LUMPSUM OR SIP)
    # --------------------------------------------------
    filtered = df[df["amc_name"] == data.amc_name]

    if data.investment_amount:
        filtered = filtered[
            (filtered["min_lumpsum"].isna()) |
            (filtered["min_lumpsum"] <= data.investment_amount)
        ]

    if data.sip_amount:
        filtered = filtered[
            (filtered["min_sip"].isna()) |
            (filtered["min_sip"] <= data.sip_amount)
        ]

    if mapped_category:
        filtered = filtered[filtered["category"] == mapped_category]

    if filtered.empty:
        return {"error": "No suitable schemes found"}

    # --------------------------------------------------
    # QUALITY FILTERS (ROBUSTNESS)
    # --------------------------------------------------
    filtered = filtered[
        (filtered["fund_age_yr"] >= 3) &
        (filtered["expense_ratio"] < 3)
    ]

    # --------------------------------------------------
    # ADDED: Clip extreme past returns to avoid outliers dominating sort
    # --------------------------------------------------
    filtered = filtered.copy()  # Avoid SettingWithCopyWarning
    filtered["returns_1yr"] = filtered["returns_1yr"].clip(upper=40)  # Realistic cap (adjustable)

    # --------------------------------------------------
    # ADDED: Smarter sorting – prioritize risk-adjusted metrics for Debt
    # --------------------------------------------------
    if mapped_category == "Debt":
        filtered = filtered.sort_values(
            by=["sharpe", "sortino", "returns_1yr"],
            ascending=False
        ).head(10)
    else:
        filtered = filtered.sort_values(
            by=["returns_1yr", "sharpe"],
            ascending=False
        ).head(10)

    recommendations = []

    # --------------------------------------------------
    # ML + MONTE CARLO LOOP
    # --------------------------------------------------
    for _, scheme in filtered.iterrows():

        input_dict = {}
        for col in preprocessor.feature_names_in_:
            if col == "tenure":
                input_dict[col] = data.tenure
            elif col in scheme:
                input_dict[col] = scheme[col]
            else:
                input_dict[col] = 0

        X = preprocessor.transform(pd.DataFrame([input_dict]))
        
        # Model prediction (works for PyTorch or TF wrapper)
        delta = float(model.predict(X)[0])

        # Base + delta (NO MULTIPLICATION)
        future_return = BASE_RETURN + delta

        # Soft confidence boost for long-term investors
        if data.tenure >= 10:
            future_return += 1.0
        elif data.tenure >= 5:
            future_return += 0.5

        future_return = np.clip(future_return, MIN_RETURN, MAX_RETURN)

        mc = monte_carlo(
            future_return,
            scheme["sd"],
            data.tenure
        )

        score = (
            0.5 * mc["expected"] +
            0.3 * scheme["returns_1yr"] +
            0.2 * scheme["sharpe"]
        )

        recommendations.append({
            "scheme_name": scheme["scheme_name"],
            "past_1yr": round(float(scheme["returns_1yr"]), 2),
            "expected": mc["expected"],
            "best_case": mc["best_case"],
            "worst_case": mc["worst_case"],
            "risk": map_risk_level(int(scheme.get("risk_level", 4))),
            "score": round(score, 2)
        })

    recommendations = sorted(
        recommendations,
        key=lambda x: x["score"],
        reverse=True
    )[:5]

    return {
        "recommendations": recommendations
    }