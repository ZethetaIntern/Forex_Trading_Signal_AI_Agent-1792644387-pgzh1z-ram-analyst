import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "Reports" / "forex_technical_indicators.csv"
OUTPUT_FILE = PROJECT_ROOT / "Reports" / "random_forest_fair_comparison_results.csv"


# 2. Load the technical indicators dataset
df = pd.read_csv(DATA_FILE)

df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)


# 3. Create the next-hour direction target
df["Next_Close"] = df["Close"].shift(-1)

df["Target"] = (
    df["Next_Close"] > df["Close"]
).astype(int)


# 4. Define the original 20 features
full_features = [
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


# 5. Define the reduced 8 features
reduced_features = [
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Std",
    "ATR_14",
    "Volume",
    "Spread",
]


# 6. Keep only records usable by BOTH models
required_columns = full_features + ["Target", "Next_Close"]

df = df.dropna(subset=required_columns).copy()
df = df.reset_index(drop=True)


# 7. Create identical chronological training and test periods
split_index = int(len(df) * 0.80)

X_full = df[full_features]
X_reduced = df[reduced_features]
y = df["Target"]

X_full_train = X_full.iloc[:split_index]
X_full_test = X_full.iloc[split_index:]

X_reduced_train = X_reduced.iloc[:split_index]
X_reduced_test = X_reduced.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# 8. Train the full 20-feature Random Forest model
full_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced",
)

full_model.fit(X_full_train, y_train)

full_predictions = full_model.predict(X_full_test)
full_accuracy = accuracy_score(y_test, full_predictions)


# 9. Train the reduced 8-feature Random Forest model
reduced_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced",
)

reduced_model.fit(X_reduced_train, y_train)

reduced_predictions = reduced_model.predict(X_reduced_test)
reduced_accuracy = accuracy_score(y_test, reduced_predictions)


# 10. Calculate a majority-class baseline
majority_class = y_train.mode()[0]

baseline_predictions = [majority_class] * len(y_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)


# 11. Print the comparison details
print("\n========== FAIR RANDOM FOREST COMPARISON ==========")

print(f"Common usable records: {len(df)}")
print(f"Training records: {len(X_full_train)}")
print(f"Testing records: {len(X_full_test)}")

print(f"Test period starts: {df['DateTime'].iloc[split_index]}")
print(f"Test period ends:   {df['DateTime'].iloc[-1]}")

print("\n--- Model Accuracies ---")
print(f"Majority-class baseline: {baseline_accuracy * 100:.2f}%")
print(f"Full model (20 features): {full_accuracy * 100:.2f}%")
print(f"Reduced model (8 features): {reduced_accuracy * 100:.2f}%")


print("\n--- Full Model Classification Report ---")
print(
    classification_report(
        y_test,
        full_predictions,
        labels=[0, 1],
        target_names=["Down_or_unchanged", "Up"],
        zero_division=0,
    )
)


print("\n--- Reduced Model Classification Report ---")
print(
    classification_report(
        y_test,
        reduced_predictions,
        labels=[0, 1],
        target_names=["Down_or_unchanged", "Up"],
        zero_division=0,
    )
)


# 12. Save the comparison results
results = pd.DataFrame(
    [
        {
            "Model": "Majority-class baseline",
            "Features": 0,
            "Training_Records": len(X_full_train),
            "Testing_Records": len(X_full_test),
            "Test_Accuracy_Percent": round(baseline_accuracy * 100, 2),
        },
        {
            "Model": "Random Forest - Full",
            "Features": len(full_features),
            "Training_Records": len(X_full_train),
            "Testing_Records": len(X_full_test),
            "Test_Accuracy_Percent": round(full_accuracy * 100, 2),
        },
        {
            "Model": "Random Forest - Reduced",
            "Features": len(reduced_features),
            "Training_Records": len(X_reduced_train),
            "Testing_Records": len(X_reduced_test),
            "Test_Accuracy_Percent": round(reduced_accuracy * 100, 2),
        },
    ]
)

results.to_csv(OUTPUT_FILE, index=False)

print("\nComparison report saved successfully:")
print(OUTPUT_FILE)

