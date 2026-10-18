
import pandas as pd
from pathlib import Path

# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "Reports"

INPUT_FILE = REPORTS_DIR / "xgboost_walk_forward_results.csv"
OUTPUT_FILE = REPORTS_DIR / "xgboost_walk_forward_summary.csv"

# 2. Check input file
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Walk-forward results not found: {INPUT_FILE}"
    )

# 3. Load results
df = pd.read_csv(INPUT_FILE)

required_columns = {
    "Fold",
    "Training_Records",
    "Testing_Records",
    "XGBoost_Accuracy_Percent",
    "Baseline_Accuracy_Percent",
    "XGBoost_Balanced_Accuracy_Percent",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

# 4. Calculate summary metrics
summary = pd.DataFrame([
    {
        "Metric": "Number of folds",
        "Value": len(df),
        "Unit": "folds",
    },
    {
        "Metric": "Mean XGBoost accuracy",
        "Value": round(df["XGBoost_Accuracy_Percent"].mean(), 2),
        "Unit": "%",
    },
    {
        "Metric": "Mean baseline accuracy",
        "Value": round(df["Baseline_Accuracy_Percent"].mean(), 2),
        "Unit": "%",
    },
    {
        "Metric": "Mean XGBoost balanced accuracy",
        "Value": round(
            df["XGBoost_Balanced_Accuracy_Percent"].mean(), 2
        ),
        "Unit": "%",
    },
    {
        "Metric": "Best fold XGBoost accuracy",
        "Value": round(df["XGBoost_Accuracy_Percent"].max(), 2),
        "Unit": "%",
    },
    {
        "Metric": "Worst fold XGBoost accuracy",
        "Value": round(df["XGBoost_Accuracy_Percent"].min(), 2),
        "Unit": "%",
    },
])

# 5. Display fold results and summary
print("\n========== WALK-FORWARD FOLD RESULTS ==========")
print(df.to_string(index=False))

print("\n========== WALK-FORWARD SUMMARY ==========")
print(summary.to_string(index=False))

# 6. Save summary
summary.to_csv(OUTPUT_FILE, index=False)

print(f"\nSummary saved successfully:\n{OUTPUT_FILE}")

print(
    "\nInterpretation note: These results come from a small dataset "
    "and are exploratory. They do not establish future performance "
    "or trading profitability."
)
