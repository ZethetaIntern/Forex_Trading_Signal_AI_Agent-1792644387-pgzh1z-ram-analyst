import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "Reports" / "forex_technical_indicators.csv"
MODEL_FILE = PROJECT_ROOT / "Reports" / "xgboost_model.joblib"
RESULTS_FILE = PROJECT_ROOT / "Reports" / "xgboost_results.csv"


# 2. Load and sort the dataset
df = pd.read_csv(DATA_FILE)

df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)


# 3. Create the next-hour price direction target
df["Next_Close"] = df["Close"].shift(-1)

df["Target"] = (
    df["Next_Close"] > df["Close"]
).astype(int)


# 4. Select the same 20 features as the Random Forest model
features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Spread",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "RSI_14",
    "EMA_12",
    "EMA_26",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Middle",
    "BB_Std",
    "BB_Upper",
    "BB_Lower",
    "ATR_14",
]


# 5. Remove records with missing required values
df = df.dropna(
    subset=features + ["Target", "Next_Close"]
).copy()

df = df.reset_index(drop=True)


# 6. Split chronologically: first 80% for training, last 20% for testing
split_index = int(len(df) * 0.80)

X = df[features]
y = df["Target"]

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# 7. Train the XGBoost classifier
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


# 8. Evaluate on unseen test records
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========== XGBOOST MODEL RESULTS ==========")
print(f"Usable records: {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Test period starts: {df['DateTime'].iloc[split_index]}")
print(f"Test period ends: {df['DateTime'].iloc[-1]}")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\n--- Classification Report ---")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Down_or_unchanged", "Up"],
        zero_division=0,
    )
)

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, predictions))


# 9. Save the trained model
dump(model, MODEL_FILE)
print(f"\nTrained model saved to: {MODEL_FILE}")


# 10. Save evaluation results
results = pd.DataFrame(
    [
        {
            "Model": "XGBoost",
            "Features": len(features),
            "Training_Records": len(X_train),
            "Testing_Records": len(X_test),
            "Test_Accuracy_Percent": round(accuracy * 100, 2),
        }
    ]
)

results.to_csv(RESULTS_FILE, index=False)

print(f"Results report saved to: {RESULTS_FILE}")