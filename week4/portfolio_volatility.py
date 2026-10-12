import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

prices = pd.read_csv("data/prices.csv", parse_dates=["Date"], index_col="Date")
log_returns = np.log(prices / prices.shift(1)).dropna()
returns_jpm = log_returns["JPM"]
returns_bac = log_returns["BAC"]


def portfolio_volatility(returns_x, returns_y, w):
  """Computes portfolio volatility (standard deviation) for a two-asset portfolio.

  Parameters:
  -----------
  returns_x : pd.Series
      Return series for asset X.
  returns_y : pd.Series
      Return series for asset Y.
  w : float
      Weight allocated to asset X (0 <= w <= 1).

  Returns:
  --------
  float
      Portfolio volatility (sigma_p).
  """
  var_x = returns_x.var()
  var_y = returns_y.var()
  cov_xy = returns_x.cov(returns_y)

  port_var = (w**2) * var_x + ((1 - w) ** 2) * var_y + 2 * w * (1 - w) * cov_xy
  return np.sqrt(port_var)

vol = portfolio_volatility(returns_jpm, returns_bac, w=0.5)
print(f"Portfolio Volatility: {vol:.6f}")

# Individual asset volatilities (std dev)
std_jpm = returns_jpm.std()
std_bac = returns_bac.std()

# Weighted average volatility (linear combination)
weighted_vol = 0.5 * std_jpm + 0.5 * std_bac

# Calculated portfolio volatility from your function
vol_p = portfolio_volatility(returns_jpm, returns_bac, w=0.5)

print(f"Portfolio Volatility (sigma_p): {vol_p:.6f}")
print(f"Weighted Average Volatility:   {weighted_vol:.6f}")
print(f"Diversification Gap:            {weighted_vol - vol_p:.6f}")

# Extract PG return series
returns_pg = log_returns["PG"]

# Compute individual standard deviations
std_jpm = returns_jpm.std()
std_pg = returns_pg.std()

# 50/50 Portfolio Volatility vs Weighted Average Volatility
vol_p_pg = portfolio_volatility(returns_jpm, returns_pg, w=0.5)
weighted_vol_pg = 0.5 * std_jpm + 0.5 * std_pg

print(f"JPM/PG Portfolio Volatility (sigma_p): {vol_p_pg:.6f}")
print(f"JPM/PG Weighted Average Volatility:   {weighted_vol_pg:.6f}")
print(
    "JPM/PG Diversification Gap:           "
    f" {weighted_vol_pg - vol_p_pg:.6f}"
)

weights = []
vols = []

# Step w explicitly from 0 to 1 with an actual for loop
for i in range(101):  # 0 to 100 inclusive
  w = i / 100.0  # Increment of 0.01
  sigma_p = portfolio_volatility(returns_jpm, returns_pg, w)

  weights.append(w)
  vols.append(sigma_p)

# 2. Plot portfolio volatility (sigma_p) against weight (w)
plt.figure(figsize=(8, 5))
plt.plot(weights, vols, label="JPM / PG Portfolio Curve", color="blue")
plt.title(
    "Portfolio Volatility vs. Weight on JPM (rho = 0.388)", fontsize=12
)  #
plt.xlabel("Weight on JPM (w)", fontsize=10)
plt.ylabel("Portfolio Volatility (sigma_p)", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()