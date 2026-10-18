
import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier

# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "Reports" / "forex_technical_indicators.csv"
OUTPUT_FILE = PROJECT_ROOT / "Reports" / "xgboost_tuning_results.csv"

# 2. Load and sort data
df = pd.read_csv(DATA_FILE)
df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)

# 3. Create next-hour direction target
df["Next_Close"] = df["Close"].shift(-1)
df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)

# 4. Use the same 20 features as the original model
features = [
    "Open", "High", "Low", "Close", "Volume", "Spread",
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "EMA_12", "EMA_26", "MACD", "MACD_Signal",
    "MACD_Histogram", "BB_Middle", "BB_Std",
    "BB_Upper", "BB_Lower", "ATR_14",
]

df = df.dropna(subset=features + ["Target", "Next_Close"])
df = df.reset_index(drop=True)

# 5. Keep the chronological split fixed
split_index = int(len(df) * 0.80)

X_train = df[features].iloc[:split_index]
X_test = df[features].iloc[split_index:]
y_train = df["Target"].iloc[:split_index]
y_test = df["Target"].iloc[split_index:]

print("Usable records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Test start:", df["DateTime"].iloc[split_index])
print("Test end:", df["DateTime"].iloc[-1])

# 6. Majority-class baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

results = [{
    "Model": "Majority-class baseline",
    "Test_Accuracy_Percent": round(baseline_accuracy * 100, 2),
}]

print(f"\nBaseline accuracy: {baseline_accuracy * 100:.2f}%")

# 7. Compare configurations on identical records
configurations = [
    {
        "name": "XGBoost original",
        "params": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },
    {
        "name": "XGBoost shallow",
        "params": {
            "n_estimators": 150,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },
    {
        "name": "XGBoost regularized",
        "params": {
            "n_estimators": 150,
            "max_depth": 2,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 5,
            "min_child_weight": 5,
        },
    },
]

for config in configurations:
    model = XGBClassifier(
        **config["params"],
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Model": config["name"],
        "Test_Accuracy_Percent": round(accuracy * 100, 2),
    })

    print(f"\n{config['name']}: {accuracy * 100:.2f}%")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Down_or_unchanged", "Up"],
        zero_division=0,
    ))

# 8. Save comparison
results_df = pd.DataFrame(results).sort_values(
    "Test_Accuracy_Percent",
    ascending=False,
).reset_index(drop=True)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\n========== TUNING RESULTS ==========")
print(results_df.to_string(index=False))
print(f"\nResults saved to:\n{OUTPUT_FILE}")
print(
    "\nNote: This is a small exploratory experiment. "
    "The test set is used to compare configurations, so a final "
    "unbiased performance estimate requires a separate untouched "
    "test period or walk-forward evaluation."
)
