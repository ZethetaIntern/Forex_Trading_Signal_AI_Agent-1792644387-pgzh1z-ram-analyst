
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "Reports" / "forex_technical_indicators.csv"
MODEL_FILE = BASE_DIR / "Reports" / "xgboost_model.joblib"
REPORT_DIR = BASE_DIR / "Reports"

FEATURES = [
    "Open", "High", "Low", "Close", "Volume", "Spread",
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "EMA_12", "EMA_26", "MACD", "MACD_Signal",
    "MACD_Histogram", "BB_Middle", "BB_Std",
    "BB_Upper", "BB_Lower", "ATR_14",
]

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load the technical-indicator dataset and trained model
df = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_FILE)

# Prepare the same chronological test split
df["Next_Close"] = df["Close"].shift(-1)
df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)
df = df.dropna(subset=FEATURES + ["Next_Close"]).reset_index(drop=True)

X = df[FEATURES]
split_index = int(len(df) * 0.8)
X_test = X.iloc[split_index:].copy()
X_explain = X_test.head(30)

if X_explain.empty:
    raise ValueError("No test records are available.")

print("Loaded trained XGBoost model.")
print("Test records explained:", len(X_explain))
print("Features:", len(FEATURES))

# Model-agnostic SHAP explanation
background = X.iloc[:50].copy()

def predict_probability(data):
    data = pd.DataFrame(data, columns=FEATURES)
    return model.predict_proba(data)[:, 1]

explainer = shap.PermutationExplainer(
    predict_probability,
    background
)

shap_values = explainer(
    X_explain,
    max_evals=2 * len(FEATURES) + 1
)

# Save summary plot
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "xgboost_shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# Save mean absolute SHAP values
importance = pd.DataFrame({
    "Feature": FEATURES,
    "Mean_Absolute_SHAP": abs(shap_values.values).mean(axis=0),
}).sort_values("Mean_Absolute_SHAP", ascending=False)

importance.to_csv(
    REPORT_DIR / "xgboost_shap_values.csv",
    index=False
)

# Save bar plot
shap.plots.bar(shap_values, max_display=20, show=False)
plt.savefig(
    REPORT_DIR / "xgboost_shap_bar.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("\nSHAP analysis completed successfully.")
print("Summary plot:", REPORT_DIR / "xgboost_shap_summary.png")
print("Bar plot:", REPORT_DIR / "xgboost_shap_bar.png")
print("SHAP values CSV:", REPORT_DIR / "xgboost_shap_values.csv")
print("\nTop 10 features:")
print(importance.head(10).to_string(index=False))
