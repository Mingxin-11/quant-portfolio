import pandas as pd
import numpy as np
prices = pd. read_csv("data/prices.csv", parse_dates=["Date"], index_col = "Date")
print(f"Number of rows: {len(prices)}")
print(f"Date range: {prices.index.min()} to {prices.index.max()}")
print("Missing values per column:", prices.isna().sum())
simple_returns = prices.pct_change()
log_returns = np.log(prices / prices.shift(1))
print("NaNs in simple returns:", simple_returns.isnull().sum())
print("NaNs in log returns:\n", log_returns.isnull().sum())
simple_returns = simple_returns.dropna()
log_returns = log_returns.dropna()
portfolio_exact = 0.5 * simple_returns["JPM"] + 0.5 * simple_returns["PG"]
portfolio_log_approx = np.exp(0.5 * log_returns["JPM"] + 0.5 * log_returns["PG"]) - 1
diff = (portfolio_exact - portfolio_log_approx).abs().max()
print(f"Maximum absolute discrepancy between Exact and Log Shortcut: {diff:.8f}")