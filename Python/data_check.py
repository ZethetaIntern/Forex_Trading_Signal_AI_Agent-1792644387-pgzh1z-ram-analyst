
import pandas as pd
from pathlib import Path

# Locate the project's data folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load the three datasets
forex_data = pd.read_csv(DATA_DIR / "forex_price_data.csv")
mc_data = pd.read_csv(DATA_DIR / "mc_scenarios.csv")
trade_data = pd.read_csv(DATA_DIR / "trade_log.csv")

# Display dataset information
print("FOREX PRICE DATA")
print("Shape:", forex_data.shape)
print(forex_data.head())
print()

print("MONTE CARLO SCENARIOS")
print("Shape:", mc_data.shape)
print(mc_data.head())
print()

print("TRADE LOG")
print("Shape:", trade_data.shape)
print(trade_data.head())
print()

print("SUCCESS: All three datasets loaded successfully!")


# Step 22: Basic data quality checks
datasets = {
    "Forex Price Data": forex_data,
    "Monte Carlo Scenarios": mc_data,
    "Trade Log": trade_data,
}

print("\n" + "=" * 50)
print("DATA QUALITY REPORT")
print("=" * 50)

for name, df in datasets.items():
    print(f"\n{name}")
    print("-" * 30)
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Column names:", list(df.columns))


# Step 23: Validate forex price data
print("\n" + "=" * 50)
print("FOREX PRICE VALIDATION")
print("=" * 50)

# Check whether High is below Low
invalid_high_low = forex_data["High"] < forex_data["Low"]

# Check whether Open or Close falls outside the High-Low range
invalid_open = (
    (forex_data["Open"] > forex_data["High"]) |
    (forex_data["Open"] < forex_data["Low"])
)

invalid_close = (
    (forex_data["Close"] > forex_data["High"]) |
    (forex_data["Close"] < forex_data["Low"])
)

print("High below Low:", invalid_high_low.sum())
print("Open outside High-Low range:", invalid_open.sum())
print("Close outside High-Low range:", invalid_close.sum())

# Check for non-positive prices
price_columns = ["Open", "High", "Low", "Close"]
print(
    "Non-positive prices:",
    (forex_data[price_columns] <= 0).sum().sum()
)


# Step 24: Check dates and currency pairs
print("\n" + "=" * 50)
print("DATE AND CURRENCY PAIR REPORT")
print("=" * 50)

# Convert DateTime to proper datetime format
forex_data["DateTime"] = pd.to_datetime(
    forex_data["DateTime"],
    errors="coerce"
)

print("Invalid dates:", forex_data["DateTime"].isna().sum())
print("Earliest date:", forex_data["DateTime"].min())
print("Latest date:", forex_data["DateTime"].max())
print("Unique currency pairs:", forex_data["PairName"].nunique())

print("\nRecords per currency pair:")
print(forex_data["PairName"].value_counts().sort_index())


# Step 25: Check chronological order and timestamps
print("\n" + "=" * 50)
print("TIME SERIES VALIDATION")
print("=" * 50)

# Check whether dates are in chronological order
print(
    "Chronologically sorted:",
    forex_data["DateTime"].is_monotonic_increasing
)

# Check duplicate timestamps within the currency pair
print(
    "Duplicate timestamps:",
    forex_data.duplicated(
        subset=["PairName", "DateTime"]
    ).sum()
)

# Check the time gaps between consecutive records
time_gaps = forex_data["DateTime"].sort_values().diff()

print("\nTime gap distribution:")
print(time_gaps.value_counts().head(10))

print(
    "\nLargest time gap:",
    time_gaps.max()
)
