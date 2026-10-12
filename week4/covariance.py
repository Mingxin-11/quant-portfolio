import pandas as pd
import numpy as np
prices = pd.read_csv("data/prices.csv", parse_dates=["Date"], index_col="Date")
log_returns = np.log(prices / prices.shift(1)).dropna()


jpm_dev = log_returns["JPM"] - log_returns["JPM"].mean()
pg_dev  = log_returns["PG"]  - log_returns["PG"].mean()
product = jpm_dev * pg_dev
n = len(log_returns)
cov_manual = product.sum() / (n - 1)
print(f"cov_manual (JPM,PG) : {cov_manual:.9f}")

cov_pandas = log_returns["JPM"].cov(log_returns["PG"])
tolerance = 1e-9
print(f"cov_pandas (JPM, PG) : {cov_pandas:.9f}")
print(f"Manual and Pandas agree: {abs(cov_manual - cov_pandas) < tolerance}")

cov_jpm_jpm = log_returns["JPM"].cov(log_returns["JPM"])
var_jpm = log_returns["JPM"].var()
print(f"\nCov(JPM, JPM): {cov_jpm_jpm:.9f}")
print(f"Var(JPM):      {var_jpm:.9f}")
print(f"Cov(X, X) == Var(X): {abs(cov_jpm_jpm - var_jpm) < tolerance}")

cov_jpm_bac = log_returns["JPM"].cov(log_returns["BAC"])
print(f"\nCov(JPM, BAC) : {cov_jpm_bac:.9f}")
print("Check Predictions: Both signs are positive as predicted, with JPM-BAC being larger.")

pg_scaled = log_returns["PG"] * 100
cov_jpm_pg_scaled = log_returns["JPM"].cov(pg_scaled)
expected_scaled_cov = 100 * cov_pandas
print(f"\nCov(JPM, PG * 100) : {cov_jpm_pg_scaled:.9f}")
print(f"100 * Cov(JPM, PG) :  {expected_scaled_cov:.9f}")
print(f"scaling identity holds: {abs(cov_jpm_pg_scaled - expected_scaled_cov) < tolerance}")
