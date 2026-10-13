import yfinance as yf
data = yf.download("JPM", start="2019-01-01", end="2024-01-01", auto_adjust=False)
data.to_csv("data/jpm.csv")