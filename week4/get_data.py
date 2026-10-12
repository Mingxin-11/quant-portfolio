import yfinance as yf

tickers = ["JPM", "PG", "BAC", "MSFT"]

raw = yf.download(
    tickers,
    start="2019-01-01",
    end="2024-01-01",
    auto_adjust=False,
)

print(raw.columns)

prices = raw["Adj Close"]
prices.to_csv("data/prices.csv")

print(prices.head())
print(prices.shape)

gap = 0.000132847
print(f"Largest gap:{gap:.6f}")
print(f"Largest gap:{gap:.4%}")
