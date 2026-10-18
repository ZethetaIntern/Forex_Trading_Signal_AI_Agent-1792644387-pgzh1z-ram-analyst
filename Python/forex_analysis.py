
import pandas as pd
from pathlib import Path

# ============================================================
# PROJECT PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "Reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "forex_price_data.csv"

# Illustrative assumption for this practice backtest.
# This is not a verified broker quote.
PIP_SIZE = 0.0001
TRANSACTION_COST_PIPS = 1.0


# ============================================================
# STEP 1: LOAD AND VALIDATE FOREX DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "DateTime", "PairName", "Open", "High", "Low", "Close"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["DateTime"] = pd.to_datetime(df["DateTime"], errors="raise")
df = df.sort_values("DateTime").reset_index(drop=True)

if df.empty:
    raise ValueError("The forex dataset is empty.")

if df["DateTime"].duplicated().any():
    raise ValueError("Duplicate timestamps found. Please check the dataset.")

print("\nFOREX DATA ANALYSIS")
print("=" * 50)
print("Total records:", len(df))
print("Currency pair:", df["PairName"].unique())
print("First date:", df["DateTime"].min())
print("Last date:", df["DateTime"].max())

print("\nPRICE SUMMARY")
print(df[["Open", "High", "Low", "Close"]].describe())


# ============================================================
# STEP 2: SIMPLE MOVING AVERAGES
# ============================================================

df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()

print("\nMOVING AVERAGE ANALYSIS")
print("=" * 50)
print(df[["DateTime", "Close", "SMA_20", "SMA_50"]].tail(10))
print("Missing SMA_20 values:", df["SMA_20"].isna().sum())
print("Missing SMA_50 values:", df["SMA_50"].isna().sum())


# ============================================================
# STEP 3: EXPONENTIAL MOVING AVERAGE
# ============================================================

df["EMA_20"] = df["Close"].ewm(
    span=20,
    adjust=False
).mean()

print("\nEXPONENTIAL MOVING AVERAGE ANALYSIS")
print("=" * 50)
print(df[["DateTime", "Close", "EMA_20"]].tail(10))


# ============================================================
# STEP 4: RELATIVE STRENGTH INDEX (RSI)
# ============================================================

delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(
    alpha=1 / 14,
    min_periods=14,
    adjust=False
).mean()

avg_loss = loss.ewm(
    alpha=1 / 14,
    min_periods=14,
    adjust=False
).mean()

rs = avg_gain / avg_loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# Handle periods where there are gains but no losses.
df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI_14"] = 100
df.loc[(avg_loss == 0) & (avg_gain == 0), "RSI_14"] = 50

print("\nRSI ANALYSIS")
print("=" * 50)
print(df[["DateTime", "Close", "RSI_14"]].tail(10))
print("Missing RSI_14 values:", df["RSI_14"].isna().sum())
print("Latest RSI_14:", round(df["RSI_14"].iloc[-1], 2))


# ============================================================
# STEP 5: MACD
# ============================================================

df["EMA_12"] = df["Close"].ewm(
    span=12,
    adjust=False
).mean()

df["EMA_26"] = df["Close"].ewm(
    span=26,
    adjust=False
).mean()

df["MACD"] = df["EMA_12"] - df["EMA_26"]

df["MACD_Signal"] = df["MACD"].ewm(
    span=9,
    adjust=False
).mean()

df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

print("\nMACD ANALYSIS")
print("=" * 50)
print(
    df[
        ["DateTime", "Close", "MACD", "MACD_Signal",
         "MACD_Histogram"]
    ].tail(10)
)

print("Latest MACD:", round(df["MACD"].iloc[-1], 6))
print("Latest MACD Signal:", round(df["MACD_Signal"].iloc[-1], 6))
print("Latest MACD Histogram:", round(df["MACD_Histogram"].iloc[-1], 6))


# ============================================================
# STEP 6: BOLLINGER BANDS
# ============================================================

df["BB_Middle"] = df["Close"].rolling(window=20).mean()
df["BB_Std"] = df["Close"].rolling(window=20).std()

df["BB_Upper"] = df["BB_Middle"] + (2 * df["BB_Std"])
df["BB_Lower"] = df["BB_Middle"] - (2 * df["BB_Std"])

print("\nBOLLINGER BANDS ANALYSIS")
print("=" * 50)
print(
    df[
        ["DateTime", "Close", "BB_Upper", "BB_Middle", "BB_Lower"]
    ].tail(10)
)


# ============================================================
# STEP 7: AVERAGE TRUE RANGE (ATR)
# ============================================================

previous_close = df["Close"].shift(1)

true_range = pd.concat(
    [
        df["High"] - df["Low"],
        (df["High"] - previous_close).abs(),
        (df["Low"] - previous_close).abs()
    ],
    axis=1
).max(axis=1)

df["ATR_14"] = true_range.ewm(
    alpha=1 / 14,
    min_periods=14,
    adjust=False
).mean()

print("\nATR ANALYSIS")
print("=" * 50)
print(df[["DateTime", "High", "Low", "Close", "ATR_14"]].tail(10))
print("Missing ATR_14 values:", df["ATR_14"].isna().sum())
print("Latest ATR_14:", round(df["ATR_14"].iloc[-1], 6))


# ============================================================
# STEP 8: CREATE TRADING SIGNALS
# ============================================================

df["Trading_Signal"] = "HOLD"

buy_condition = (
    (df["EMA_20"] > df["SMA_20"]) &
    (df["MACD"] > df["MACD_Signal"]) &
    (df["RSI_14"] > 50)
)

sell_condition = (
    (df["EMA_20"] < df["SMA_20"]) &
    (df["MACD"] < df["MACD_Signal"]) &
    (df["RSI_14"] < 50)
)

df.loc[buy_condition, "Trading_Signal"] = "BUY"
df.loc[sell_condition, "Trading_Signal"] = "SELL"

print("\nTRADING SIGNAL ANALYSIS")
print("=" * 50)
print(df["Trading_Signal"].value_counts())

print("\nLATEST TRADING SIGNALS")
print(
    df[
        ["DateTime", "Close", "EMA_20", "SMA_20", "RSI_14",
         "MACD", "MACD_Signal", "Trading_Signal"]
    ].tail(10)
)


# ============================================================
# STEP 9: EXPORT INDICATORS AND SIGNALS
# ============================================================

indicator_file = OUTPUT_DIR / "forex_technical_indicators.csv"
signals_file = OUTPUT_DIR / "forex_trading_signals.csv"

df.to_csv(indicator_file, index=False)
df.to_csv(signals_file, index=False)

print("\nDATA EXPORT")
print("=" * 50)
print("Technical indicators saved:", indicator_file)
print("Trading signals saved:", signals_file)
print("Rows exported:", len(df))


# ============================================================
# STEP 10: NEXT-CANDLE BACKTEST
# Signal is generated at candle t close.
# Entry assumption: candle t+1 open.
# Exit assumption: candle t+1 close.
# ============================================================

df["Next_Open"] = df["Open"].shift(-1)
df["Next_Close"] = df["Close"].shift(-1)

# Price return from next candle open to its close.
df["Signal_Return"] = (
    (df["Next_Close"] - df["Next_Open"]) / df["Next_Open"]
)

df["Strategy_Return"] = 0.0

df.loc[
    df["Trading_Signal"] == "BUY",
    "Strategy_Return"
] = df["Signal_Return"]

df.loc[
    df["Trading_Signal"] == "SELL",
    "Strategy_Return"
] = -df["Signal_Return"]

# The final row has no next candle, so it cannot be tested.
backtest = df.dropna(
    subset=["Next_Open", "Next_Close"]
).copy()

backtest["Cumulative_Strategy_Return"] = (
    (1 + backtest["Strategy_Return"]).cumprod() - 1
)

print("\nNEXT-CANDLE BACKTEST RESULTS")
print("=" * 50)
print("Backtest records:", len(backtest))
print(
    "Active signal periods:",
    (backtest["Trading_Signal"] != "HOLD").sum()
)
print(
    "Winning active periods:",
    (
        (backtest["Trading_Signal"] != "HOLD") &
        (backtest["Strategy_Return"] > 0)
    ).sum()
)
print(
    "Losing active periods:",
    (
        (backtest["Trading_Signal"] != "HOLD") &
        (backtest["Strategy_Return"] < 0)
    ).sum()
)
print(
    "Gross cumulative return (%):",
    round(backtest["Cumulative_Strategy_Return"].iloc[-1] * 100, 4)
)


# ============================================================
# STEP 11: APPLY ILLUSTRATIVE TRANSACTION COST
# One pip is deducted per active signal period.
# This is a simplified placeholder assumption.
# ============================================================

backtest["Transaction_Cost_Return"] = (
    TRANSACTION_COST_PIPS * PIP_SIZE / backtest["Next_Open"]
)

active_mask = backtest["Trading_Signal"].isin(["BUY", "SELL"])

backtest["Net_Strategy_Return"] = backtest["Strategy_Return"]

backtest.loc[active_mask, "Net_Strategy_Return"] = (
    backtest.loc[active_mask, "Strategy_Return"]
    - backtest.loc[active_mask, "Transaction_Cost_Return"]
)

backtest["Cumulative_Net_Return"] = (
    (1 + backtest["Net_Strategy_Return"]).cumprod() - 1
)

print("\nBACKTEST WITH TRANSACTION COSTS")
print("=" * 50)
print("Assumed transaction cost (pips):", TRANSACTION_COST_PIPS)
print(
    "Gross cumulative return (%):",
    round(backtest["Cumulative_Strategy_Return"].iloc[-1] * 100, 4)
)
print(
    "Net cumulative return after costs (%):",
    round(backtest["Cumulative_Net_Return"].iloc[-1] * 100, 4)
)


# ============================================================
# STEP 12: PERFORMANCE METRICS
# Metrics below count active signal periods, not complete trades.
# ============================================================

active_periods = backtest.loc[active_mask].copy()

print("\nADDITIONAL PERFORMANCE METRICS")
print("=" * 50)
print("Active signal periods:", len(active_periods))

if len(active_periods) > 0:
    win_rate = (
        (active_periods["Strategy_Return"] > 0).sum()
        / len(active_periods)
    ) * 100

    average_gross_return = active_periods["Strategy_Return"].mean() * 100
    average_net_return = active_periods["Net_Strategy_Return"].mean() * 100

    print("Gross win rate (%):", round(win_rate, 2))
    print("Average gross return per active period (%):",
          round(average_gross_return, 5))
    print("Average net return per active period (%):",
          round(average_net_return, 5))
else:
    print("No active BUY or SELL signal periods were found.")


# ============================================================
# STEP 13: SAVE FINAL BACKTEST RESULTS
# ============================================================

backtest_file = OUTPUT_DIR / "forex_backtest_results.csv"
backtest.to_csv(backtest_file, index=False)

print("\nBACKTEST EXPORT")
print("=" * 50)
print("Backtest file saved:", backtest_file)
print("Rows exported:", len(backtest))
print("Columns exported:", len(backtest.columns))
print("\nAnalysis completed successfully.")
