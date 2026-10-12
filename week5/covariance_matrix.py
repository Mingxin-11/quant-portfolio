import numpy as np
import pandas as pd

# 1. Load price data fresh and compute simple returns (.pct_change())
prices = pd.read_csv("data/prices.csv", parse_dates=["Date"], index_col="Date")
tickers = ["JPM", "BAC", "PG", "MSFT"]
returns = prices[tickers].pct_change().dropna()

# 2. Build the variance-covariance matrix Sigma
Sigma = returns.cov()

# 3. Extract shape and separate components
n = Sigma.shape[0]
variances = np.diag(Sigma)

# Off-diagonal boolean mask for covariances
mask = ~np.eye(n, dtype=bool)
covariances = Sigma.values[mask]

# 4. Generalized N-asset portfolio volatility function
def portfolio_volatility(weights, cov_matrix):
    """Computes portfolio volatility for any N assets using matrix algebra (sqrt(w' * Sigma * w))."""
    port_var = weights @ cov_matrix @ weights
    return np.sqrt(port_var)

# 5. Sanity Check: Test on 2 assets (JPM & BAC) against Week 4 logic
def portfolio_volatility_week4(returns_x, returns_y, w):
    var_x = returns_x.var()
    var_y = returns_y.var()
    cov_xy = returns_x.cov(returns_y)
    port_var = (w**2) * var_x + ((1 - w)**2) * var_y + 2 * w * (1 - w) * cov_xy
    return np.sqrt(port_var)

# Inputs for equal-weighted JPM/BAC pair
w_2asset = np.array([0.5, 0.5])
Sigma_2asset = returns[["JPM", "BAC"]].cov()

vol_week4 = portfolio_volatility_week4(returns["JPM"], returns["BAC"], 0.5)
vol_matrix_2asset = portfolio_volatility(w_2asset, Sigma_2asset)

print("--- 2-Asset Verification ---")
print(f"Week 4 Volatility:  {vol_week4:.6f}")
print(f"Matrix Volatility:  {vol_matrix_2asset:.6f}")
print(f"Agreement Check:    {np.isclose(vol_week4, vol_matrix_2asset)}")

# 6. Run generalized function on all 4 assets (equal weights)
w_4asset = np.array([0.25, 0.25, 0.25, 0.25])
vol_4asset = portfolio_volatility(w_4asset, Sigma)

print("\n--- 4-Asset Portfolio ---")
print(f"4-Asset Volatility: {vol_4asset:.6f}")
