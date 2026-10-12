import numpy as np
import pandas as pd

# Load price data and compute simple returns R and log returns
prices = pd.read_csv("data/prices.csv", parse_dates=["Date"], index_col="Date")
returns = prices.pct_change().dropna()  # Simple returns R
log_returns = np.log(prices / prices.shift(1)).dropna()  # Log returns ln(1 + R)


def portfolio_volatility(returns_x, returns_y, w):
  """Computes portfolio volatility (sigma_p) for a two-asset portfolio."""
  var_x = returns_x.var()
  var_y = returns_y.var()
  cov_xy = returns_x.cov(returns_y)
  port_var = (w**2) * var_x + ((1 - w) ** 2) * var_y + 2 * w * (1 - w) * cov_xy
  return np.sqrt(port_var)


print("=" * 65)
print("LESSON 4 CONSOLIDATION: PORTFOLIO RISK & VOLATILITY DRAG")
print("=" * 65)

# -----------------------------------------------------------------------------
# SECTION 1: TWO-ASSET VOLATILITY & DYNAMIC CORRELATION IMPACT
# -----------------------------------------------------------------------------
print("\n--- SECTION 1: Dynamic Correlation & Diversification Gap ---")

jpm_ret = returns["JPM"]
bac_ret = returns["BAC"]
pg_ret = returns["PG"]

# Compute fresh correlation directly from the dataset
rho_jpm_bac = jpm_ret.corr(bac_ret)
rho_jpm_pg = jpm_ret.corr(pg_ret)

# Compute equal-weighted portfolio volatilities (w = 0.5)
vol_jpm = jpm_ret.std()
vol_bac = bac_ret.std()
vol_pg = pg_ret.std()

vol_p_bac = portfolio_volatility(jpm_ret, bac_ret, w=0.5)
vol_p_pg = portfolio_volatility(jpm_ret, pg_ret, w=0.5)

gap_bac = (0.5 * vol_jpm + 0.5 * vol_bac) - vol_p_bac
gap_pg = (0.5 * vol_jpm + 0.5 * vol_pg) - vol_p_pg

print(f"rho(JPM, BAC) [Empirical]: {rho_jpm_bac:.6f}")
print(f"JPM/BAC Diversification Gap: {gap_bac:.6f}")
print(f"rho(JPM, PG)  [Empirical]: {rho_jpm_pg:.6f}")
print(f"JPM/PG Diversification Gap:  {gap_pg:.6f}")

assert gap_pg > gap_bac, (
    "Monotonicity error: Lower correlation must yield a larger gap!"
)

# -----------------------------------------------------------------------------
# SECTION 2: CLOSED-FORM MINIMUM VARIANCE PORTFOLIO (MVP)
# -----------------------------------------------------------------------------
print("\n--- SECTION 2: Closed-Form MVP Weight Verification ---")

var_jpm = jpm_ret.var()
var_pg = pg_ret.var()
cov_jpm_pg = jpm_ret.cov(pg_ret)

# Analytical optimal weight formula for JPM
w_star = (var_pg - cov_jpm_pg) / (var_jpm + var_pg - 2 * cov_jpm_pg)
min_vol_analytical = portfolio_volatility(jpm_ret, pg_ret, w_star)

print(f"Analytical Optimal Weight on JPM (w*): {w_star:.6f}")
print(f"Minimum Achievable Volatility (sigma_p,min): {min_vol_analytical:.6f}")
print(f"Individual Volatilities -> JPM: {vol_jpm:.6f}, PG: {vol_pg:.6f}")

assert min_vol_analytical < min(vol_jpm, vol_pg), (
    "MVP error: Portfolio min volatility must be lower than both assets!"
)

# -----------------------------------------------------------------------------
# SECTION 3: ASYMPTOTIC SYSTEMATIC RISK FLOOR
# -----------------------------------------------------------------------------
print("\n--- SECTION 3: Multi-Asset Asymptotic Risk Floor ---")

cov_matrix = returns.cov()
n_assets = len(returns.columns)

# Calculate average individual variance and average pairwise covariance
diag_vars = np.diag(cov_matrix)
off_diag_covs = cov_matrix.values[~np.eye(n_assets, dtype=bool)]

avg_var = np.mean(diag_vars)
avg_cov = np.mean(off_diag_covs)

print(f"Total Assets (N): {n_assets}")
print(f"Average Asset Variance (bar_sigma^2):   {avg_var:.6f}")
print(f"Average Pairwise Covariance (bar_cov):  {avg_cov:.6f}")
print(
    "Systematic Risk Floor (sqrt(bar_cov)): " f" {np.sqrt(avg_cov):.6f}"
)

# Simulate portfolio variance scaling across N assets
print("\n  N Assets | Portfolio Variance | Systematic Floor")
print("  ------------------------------------------------")
for n in range(1, n_assets + 1):
  port_var_n = (1 / n) * avg_var + (1 - 1 / n) * avg_cov
  print(f"  {n:8d} | {port_var_n:18.6f} | {avg_cov:16.6f}")

# -----------------------------------------------------------------------------
# SECTION 4: EMPIRICAL VOLATILITY DRAG & JENSEN'S INEQUALITY
# -----------------------------------------------------------------------------
print("\n--- SECTION 4: Empirical Volatility Drag & Jensen's Inequality ---")

# Evaluate JPM simple returns
mu_jpm = jpm_ret.mean()
var_jpm_ret = jpm_ret.var()

# Empirical log-return E[ln(1 + R)] directly from log_returns
mean_log_return = log_returns["JPM"].mean()

# Second-order Taylor approximation target: mu - (sigma^2 / 2)
taylor_target = mu_jpm - (var_jpm_ret / 2.0)

# True compounded geometric rate g = exp(E[ln(1 + R)]) - 1
g_realized = np.exp(mean_log_return) - 1

print(f"Arithmetic Mean Return (mu):                {mu_jpm:.6f}")
print(f"Variance Penalty (sigma^2 / 2):            {var_jpm_ret / 2.0:.6f}")
print(f"Taylor Target (mu - sigma^2 / 2):           {taylor_target:.6f}")
print(f"Empirical Mean Log Return E[ln(1 + R)]:     {mean_log_return:.6f}")
print(f"Realized Geometric Growth Rate (g):        {g_realized:.6f}")

# Cross-check accuracy of the Taylor expansion
abs_diff = abs(mean_log_return - taylor_target)
print(f"Absolute Gap |E[ln(1+R)] - (mu - sigma^2/2)|: {abs_diff:.8f}")

assert abs_diff < 1e-4, "Taylor approximation error is larger than expected!"
print("\nAll consolidation checks passed successfully.")
