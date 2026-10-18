
import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "Reports" / "forex_technical_indicators.csv"
OUTPUT_FILE = PROJECT_ROOT / "Reports" / "xgboost_walk_forward_results.csv"

# 2. Load and sort chronologically
df = pd.read_csv(DATA_FILE)
df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)

# 3. Create next-hour direction target
df["Next_Close"] = df["Close"].shift(-1)
df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)

features = [
    "Open", "High", "Low", "Close", "Volume", "Spread",
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "EMA_12", "EMA_26", "MACD", "MACD_Signal",
    "MACD_Histogram", "BB_Middle", "BB_Std",
    "BB_Upper", "BB_Lower", "ATR_14",
]

df = df.dropna(subset=features + ["Next_Close"]).reset_index(drop=True)

# 4. Define expanding-window walk-forward evaluation
# First 60% is initial training data; the remaining 40% is
# divided into four chronological evaluation blocks.
initial_train_size = int(len(df) * 0.60)
remaining = len(df) - initial_train_size
block_size = remaining // 4

results = []

print("Usable records:", len(df))
print("Initial training records:", initial_train_size)
print("Evaluation blocks: 4")

for fold in range(4):
    test_start = initial_train_size + fold * block_size

    if fold == 3:
        test_end = len(df)
    else:
        test_end = initial_train_size + (fold + 1) * block_size

    train_df = df.iloc[:test_start]
    test_df = df.iloc[test_start:test_end]

    X_train = train_df[features]
    y_train = train_df["Target"]
    X_test = test_df[features]
    y_test = test_df["Target"]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    model_accuracy = accuracy_score(y_test, predictions)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    model_balanced_accuracy = balanced_accuracy_score(y_test, predictions)

    results.append({
        "Fold": fold + 1,
        "Training_Records": len(train_df),
        "Testing_Records": len(test_df),
        "Test_Start": test_df["DateTime"].iloc[0],
        "Test_End": test_df["DateTime"].iloc[-1],
        "XGBoost_Accuracy_Percent": round(model_accuracy * 100, 2),
        "Baseline_Accuracy_Percent": round(baseline_accuracy * 100, 2),
        "XGBoost_Balanced_Accuracy_Percent": round(
            model_balanced_accuracy * 100, 2
        ),
    })

    print(f"\nFold {fold + 1}")
    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))
    print("Test period:", test_df["DateTime"].iloc[0],
          "to", test_df["DateTime"].iloc[-1])
    print(f"XGBoost accuracy: {model_accuracy * 100:.2f}%")
    print(f"Baseline accuracy: {baseline_accuracy * 100:.2f}%")
    print(
        "XGBoost balanced accuracy: "
        f"{model_balanced_accuracy * 100:.2f}%"
    )

# 5. Save fold-level results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\n========== WALK-FORWARD SUMMARY ==========")
print(results_df.to_string(index=False))

print(
    "\nMean XGBoost accuracy: "
    f"{results_df['XGBoost_Accuracy_Percent'].mean():.2f}%"
)
print(
    "Mean baseline accuracy: "
    f"{results_df['Baseline_Accuracy_Percent'].mean():.2f}%"
)
print(
    "Mean XGBoost balanced accuracy: "
    f"{results_df['XGBoost_Balanced_Accuracy_Percent'].mean():.2f}%"
)

print(f"\nResults saved to:\n{OUTPUT_FILE}")
print(
    "\nNote: These are exploratory results on a small dataset. "
    "Walk-forward evaluation is more informative than one split, "
    "but does not establish trading profitability."
)
