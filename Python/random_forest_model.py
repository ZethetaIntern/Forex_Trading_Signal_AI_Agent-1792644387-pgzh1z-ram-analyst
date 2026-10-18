import pandas as pd

# Load the existing technical-indicator dataset
DATA_PATH = "Reports/forex_technical_indicators.csv"

df = pd.read_csv(DATA_PATH)
df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)

# Define the next-hour price direction:
# 1 = next close is higher; 0 = next close is equal or lower
df["Next_Close"] = df["Close"].shift(-1)
df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)

# Remove the final row because its next close is unavailable
df = df.dropna(subset=["Next_Close"]).copy()
df["Target"] = df["Target"].astype(int)

# Use only information available at the current timestamp.
# Exclude raw future prices, existing rule-based signals and identifiers.
feature_columns = [
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

# Keep rows where all selected indicators are available
df = df.dropna(subset=feature_columns + ["Target"]).copy()

print("Dataset prepared successfully.")
print("Usable records:", len(df))
print("Features:", len(feature_columns))
print("\nTarget distribution:")
print(df["Target"].value_counts().sort_index())
print("\nFeature columns:")
print(feature_columns)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Sort by time before splitting
df = df.sort_values("DateTime").reset_index(drop=True)

# Use the first 80% for training and the last 20% for testing
split_index = int(len(df) * 0.80)

X = df[feature_columns]
y = df["Target"]

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nChronological split:")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Train the Random Forest classifier
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced",
)

rf_model.fit(X_train, y_train)

# Evaluate only on the later, held-out records
y_pred = rf_model.predict(X_test)

print("\nRANDOM FOREST RESULTS")
print("-" * 35)
print("Test accuracy:", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Down_or_unchanged", "Up"],
        zero_division=0,
    )
)

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


import joblib

MODEL_PATH = "Reports/random_forest_model.joblib"
joblib.dump(rf_model, MODEL_PATH)

print("\nModel saved successfully to:", MODEL_PATH)


importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": rf_model.feature_importances_,
})

importance_df = importance_df.sort_values(
    "Importance", ascending=False
)

print("\nTOP 10 FEATURE IMPORTANCES")
print(importance_df.head(10).to_string(index=False))


importance_df.to_csv(
    "Reports/random_forest_feature_importance.csv",
    index=False,
)

print(
    "\nFeature-importance report saved to: "
    "Reports/random_forest_feature_importance.csv"
)


baseline_accuracy = 47.78
rf_accuracy = accuracy_score(y_test, y_pred) * 100

comparison_df = pd.DataFrame({
    "Model": ["Majority-class baseline", "Random Forest"],
    "Test_Accuracy_Percent": [
        baseline_accuracy,
        round(rf_accuracy, 2),
    ],
})

comparison_df.to_csv(
    "Reports/random_forest_baseline_comparison.csv",
    index=False,
)

print("\nBaseline comparison saved to:")
print("Reports/random_forest_baseline_comparison.csv")
print(comparison_df.to_string(index=False))

