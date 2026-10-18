
import pandas as pd
from pathlib import Path

# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "Reports"

FAIR_COMPARISON_FILE = (
    REPORTS_DIR / "random_forest_fair_comparison_results.csv"
)
XGBOOST_FILE = REPORTS_DIR / "xgboost_results.csv"
LSTM_FILE = REPORTS_DIR / "lstm_model_summary.csv"

OUTPUT_FILE = REPORTS_DIR / "model_comparison_report.csv"

# 2. Validate required files
for file_path in [
    FAIR_COMPARISON_FILE,
    XGBOOST_FILE,
    LSTM_FILE,
]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required report not found: {file_path}"
        )

# 3. Load Random Forest and baseline results
fair_data = pd.read_csv(FAIR_COMPARISON_FILE)

required_columns = {
    "Model",
    "Features",
    "Test_Accuracy_Percent",
}

if not required_columns.issubset(fair_data.columns):
    raise ValueError(
        "The Random Forest comparison report is missing required "
        f"columns. Found: {list(fair_data.columns)}"
    )

model_name_mapping = {
    "Majority-class baseline": "Majority-class baseline",
    "Random Forest - Full": "Random Forest - 20 features",
    "Random Forest - Reduced": "Random Forest - 8 features",
}

all_results = []

for _, row in fair_data.iterrows():
    original_name = str(row["Model"])

    if original_name in model_name_mapping:
        all_results.append({
            "Model": model_name_mapping[original_name],
            "Features": int(row["Features"]),
            "Test_Accuracy_Percent": float(
                row["Test_Accuracy_Percent"]
            ),
        })

# 4. Load XGBoost result
xgboost_data = pd.read_csv(XGBOOST_FILE)

if "Test_Accuracy_Percent" not in xgboost_data.columns:
    raise ValueError(
        "XGBoost results are missing Test_Accuracy_Percent."
    )

all_results.append({
    "Model": "XGBoost - 20 features",
    "Features": 20,
    "Test_Accuracy_Percent": float(
        xgboost_data.iloc[0]["Test_Accuracy_Percent"]
    ),
})

# 5. Load LSTM result
lstm_data = pd.read_csv(LSTM_FILE)

if "Test_Accuracy" not in lstm_data.columns:
    raise ValueError(
        "LSTM summary is missing the Test_Accuracy column."
    )

lstm_accuracy = float(lstm_data.iloc[0]["Test_Accuracy"])

all_results.append({
    "Model": "LSTM - 20 features",
    "Features": int(lstm_data.iloc[0]["Features"]),
    "Test_Accuracy_Percent": lstm_accuracy * 100,
})

# 6. Consolidate and sort results
comparison = pd.DataFrame(all_results)

comparison = comparison.drop_duplicates(
    subset=["Model"],
    keep="last",
)

comparison = comparison.sort_values(
    "Test_Accuracy_Percent",
    ascending=False,
).reset_index(drop=True)

# 7. Display comparison
print("\n========== CONSOLIDATED MODEL COMPARISON ==========")
print(comparison.to_string(index=False, formatters={
    "Test_Accuracy_Percent": "{:.2f}%".format
}))

best_model = comparison.iloc[0]

print(
    f"\nHighest reported accuracy: {best_model['Model']} "
    f"({best_model['Test_Accuracy_Percent']:.2f}%)"
)

print(
    "\nProject target: greater than 55% test accuracy."
)

if best_model["Test_Accuracy_Percent"] > 55:
    print("The accuracy target has been exceeded.")
else:
    print("The accuracy target has not yet been achieved.")

print(
    "\nNote: The LSTM uses 24-hour sequences, while the other "
    "models use individual rows. The test periods are chronological, "
    "but the evaluation setups differ. Treat this as a preliminary "
    "comparison, not a definitive model ranking."
)

# 8. Save report
comparison.to_csv(OUTPUT_FILE, index=False)

print(f"\nReport saved successfully:\n{OUTPUT_FILE}")
