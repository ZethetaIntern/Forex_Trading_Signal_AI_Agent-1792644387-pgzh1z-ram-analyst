
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the existing technical-indicator data
df = pd.read_csv("Reports/forex_technical_indicators.csv")
df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)

# Predict whether the next close is higher than the current close
df["Next_Close"] = df["Close"].shift(-1)
df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)

# Features retained for this experiment
feature_columns = [
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Std",
    "ATR_14",
    "Volume",
    "Spread",
]

df = df.dropna(subset=feature_columns + ["Next_Close"]).copy()

X = df[feature_columns]
y = df["Target"]

# Chronological 80/20 split
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Reduced-feature experiment")
print("Usable records:", len(df))
print("Features:", len(feature_columns))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100

print("\nReduced-feature test accuracy:", f"{accuracy:.2f}%")
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

# Save experiment results separately
pd.DataFrame({
    "Model": ["Random Forest - Reduced Features"],
    "Test_Accuracy_Percent": [round(accuracy, 2)],
    "Training_Records": [len(X_train)],
    "Testing_Records": [len(X_test)],
    "Feature_Count": [len(feature_columns)],
}).to_csv(
    "Reports/random_forest_reduced_features_results.csv",
    index=False,
)

print("\nResults saved to:")
print("Reports/random_forest_reduced_features_results.csv")
