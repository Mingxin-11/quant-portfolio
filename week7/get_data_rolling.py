import pandas as pd
import yfinance as yf

# Project asset universe order
tickers = ["JPM", "BAC", "PG", "MSFT"]

# Download data: start 2007-12-01, end 2026-10-01 (exclusive, last row 2026-09-30)
df = yf.download(
    tickers,
    start="2007-12-01",
    end="2026-10-01",
    auto_adjust=False,
    progress=False,
)["Adj Close"]

# Reorder columns to project standard
prices = df[tickers]

# Save to distinct rolling prices file
prices.to_csv("data/prices_rolling.csv")

# Print diagnostic metrics
print("(a) Shape:", prices.shape)
print("    First date:", prices.index[0].strftime("%Y-%m-%d"))
print("    Last date:", prices.index[-1].strftime("%Y-%m-%d"))
print("\n(b) NaNs per ticker:")
print(prices.isna().sum())
print("    Repeated dates:", prices.index.duplicated().sum())
print("\n(c) Dates strictly increasing:", prices.index.is_monotonic_increasing)
print("    Columns list:", list(prices.columns))