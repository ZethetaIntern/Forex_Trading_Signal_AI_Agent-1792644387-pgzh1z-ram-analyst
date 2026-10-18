
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import set_random_seed

# Reproducibility
set_random_seed(42)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "Reports" / "forex_technical_indicators.csv"
REPORT_DIR = BASE_DIR / "Reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Input features
FEATURES = [
    "Open", "High", "Low", "Close", "Volume", "Spread",
    "SMA_20", "SMA_50", "EMA_20", "RSI_14",
    "EMA_12", "EMA_26", "MACD", "MACD_Signal",
    "MACD_Histogram", "BB_Middle", "BB_Std",
    "BB_Upper", "BB_Lower", "ATR_14",
]

LOOKBACK = 24

# Load data
df = pd.read_csv(DATA_FILE)

# Create next-hour direction target
df["Next_Close"] = df["Close"].shift(-1)
df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)

# Retain rows with complete features and target
df = df.dropna(subset=FEATURES + ["Next_Close"]).reset_index(drop=True)

X = df[FEATURES].astype("float32").to_numpy()
y = df["Target"].astype("int32").to_numpy()

if len(df) <= LOOKBACK + 10:
    raise ValueError("Not enough usable records to build LSTM sequences.")

# Chronological split: do not shuffle
split_index = int(len(df) * 0.8)

# Fit scaler only on training-period feature rows
scaler = StandardScaler()
scaler.fit(X[:split_index])

X_scaled = scaler.transform(X).astype("float32")

# Create sequences; each target corresponds to the end of its input window
X_sequences = []
y_sequences = []
target_indices = []

for end_idx in range(LOOKBACK - 1, len(df)):
    X_sequences.append(X_scaled[end_idx - LOOKBACK + 1:end_idx + 1])
    y_sequences.append(y[end_idx])
    target_indices.append(end_idx)

X_sequences = np.asarray(X_sequences, dtype="float32")
y_sequences = np.asarray(y_sequences, dtype="int32")
target_indices = np.asarray(target_indices)

# Split by target timestamp, preserving chronological order
train_mask = target_indices < split_index
test_mask = target_indices >= split_index

X_train = X_sequences[train_mask]
y_train = y_sequences[train_mask]
X_test = X_sequences[test_mask]
y_test = y_sequences[test_mask]

print("Usable rows:", len(df))
print("Lookback hours:", LOOKBACK)
print("Training sequences:", len(X_train))
print("Testing sequences:", len(X_test))
print("Features:", len(FEATURES))

# Define LSTM classifier
model = Sequential([
    Input(shape=(LOOKBACK, len(FEATURES))),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Use the end of the training period for validation
validation_split = 0.15

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_split=validation_split,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1,
)

# Evaluate on the untouched chronological test period
probabilities = model.predict(X_test, verbose=0).ravel()
predictions = (probabilities >= 0.5).astype("int32")

accuracy = accuracy_score(y_test, predictions)

print("\nLSTM test accuracy:", f"{accuracy:.2%}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# Save model and test results
model.save(REPORT_DIR / "lstm_model.keras")
joblib.dump(scaler, REPORT_DIR / "lstm_scaler.joblib")

results = pd.DataFrame({
    "Target_Index": target_indices[test_mask],
    "Actual": y_test,
    "Predicted": predictions,
    "Predicted_Probability_Up": probabilities,
})

results.to_csv(REPORT_DIR / "lstm_results.csv", index=False)

summary = pd.DataFrame([{
    "Model": "LSTM",
    "Lookback_Hours": LOOKBACK,
    "Features": len(FEATURES),
    "Training_Sequences": len(X_train),
    "Testing_Sequences": len(X_test),
    "Test_Accuracy": accuracy,
}])

summary.to_csv(REPORT_DIR / "lstm_model_summary.csv", index=False)

# Save training history plot
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy loss")
plt.title("LSTM Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "lstm_training_loss.png", dpi=150)
plt.close()

print("\nLSTM training completed.")
print("Saved: Reports/lstm_model.keras")
print("Saved: Reports/lstm_scaler.joblib")
print("Saved: Reports/lstm_results.csv")
print("Saved: Reports/lstm_model_summary.csv")
print("Saved: Reports/lstm_training_loss.png")
