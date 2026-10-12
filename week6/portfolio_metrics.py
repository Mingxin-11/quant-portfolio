import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load prices and compute simple daily returns
prices = pd.read_csv("data/prices.csv", parse_dates=["Date"], index_col="Date")
tickers = ["JPM", "BAC", "PG", "MSFT"]
returns = prices[tickers].pct_change().dropna()

# -----------------------------------------------------------------------------
# 1. SETUP: Covariance, Mean Returns, & Functions
# -----------------------------------------------------------------------------
Sigma = returns.cov()
mean_returns = returns.mean()

# Daily risk-free rate assumption (set to 0.0 per Lesson 6.2 decision)


def portfolio_volatility(weights, cov_matrix):
  """Computes portfolio volatility using matrix algebra: sqrt(w' * Sigma * w)."""
  port_var = weights @ cov_matrix @ weights
  return np.sqrt(port_var)


def portfolio_return(weights, mean_returns):
  """Computes expected portfolio return using dot product: w' * mu."""
  return weights @ mean_returns

def sharpe_ratio(portfolio_ret, portfolio_vol, risk_free_rate):
  """Computes Sharpe ratio: (E[Rp] - rf) / sigma_p."""
  return (portfolio_ret - risk_free_rate) / portfolio_vol


# -----------------------------------------------------------------------------
# 2. CORNER TEST: 100% JPM ([1.0, 0.0, 0.0, 0.0])
# -----------------------------------------------------------------------------
w_jpm = np.array([1.0, 0.0, 0.0, 0.0])

test_ret = portfolio_return(w_jpm, mean_returns)
test_vol = portfolio_volatility(w_jpm, Sigma)
test_sharpe = sharpe_ratio(test_ret, test_vol, 0.0)

jpm_mean_direct = returns["JPM"].mean()
jpm_std_direct = returns["JPM"].std()

print("=== 2. CORNER TEST (100% JPM) ===")
print(f"Function Return:     {test_ret:.6f} | Direct pandas mean: {jpm_mean_direct:.6f}")
print(f"Function Volatility: {test_vol:.6f} | Direct pandas std:  {jpm_std_direct:.6f}")
print(f"Calculated Sharpe:   {test_sharpe:.6f}")
print(f"Previous Hand Estimates: (sigma=0.020818, E[Rp]=0.001090) -> Exact Match: {np.isclose(test_ret, 0.001090) and np.isclose(test_vol, 0.020818)}\n")


# -----------------------------------------------------------------------------
# 3. RANDOM WEIGHTS: Single Draw Inspection
# -----------------------------------------------------------------------------
print("=== 3. RANDOM WEIGHTS GENERATION (Seed 42) ===")
rng = np.random.default_rng(seed=42)
raw_draws = rng.random(4)
weights = raw_draws / raw_draws.sum()

print(f"Raw draws:   {np.round(raw_draws, 3)}")
print(f"Weights:     {np.round(weights, 3)}")
print(f"Weights Sum: {weights.sum():.6f}\n")


# -----------------------------------------------------------------------------
# 4. THE LOOP: Generating 5,000 Random Portfolios
# -----------------------------------------------------------------------------
n_portfolios = 5000

# Re-initialize RNG once before the loop so execution remains deterministic
rng = np.random.default_rng(seed=42)

all_weights = []
all_returns = []
all_volatilities = []
all_sharpes = []

for _ in range(n_portfolios):
  raw_draws = rng.random(4)
  w = raw_draws / raw_draws.sum()

  p_ret = portfolio_return(w, mean_returns)
  p_vol = portfolio_volatility(w, Sigma)
  p_sharpe = sharpe_ratio(p_ret, p_vol, 0.0)

  all_weights.append(w)
  all_returns.append(p_ret)
  all_volatilities.append(p_vol)
  all_sharpes.append(p_sharpe)

weights_array = np.array(all_weights)
returns_array = np.array(all_returns)
volatilities_array = np.array(all_volatilities)
sharpes_array = np.array(all_sharpes)


# -----------------------------------------------------------------------------
# 5. CHECKS: Verifying Array Properties
# -----------------------------------------------------------------------------
print("=== 5. CHECKS ===")
print(f"Weights Array Shape:                   {weights_array.shape}")
print(f"All Weights Sum to 1.0 (row-wise):     {np.allclose(weights_array.sum(axis=1), 1.0)}")
print(f"All Weights Non-Negative (Long-Only):  {weights_array.min() >= 0}")

# Check that all generated portfolios are distinct
unique_rows = np.unique(weights_array, axis=0)
print(len(unique_rows) == n_portfolios)


print("Average weight per asset:", weights_array.mean(axis=0))
print("Average portfolio return:", returns_array.mean())

# Create figure
plt.figure(figsize=(10, 6))

# 1. Plot simulated portfolio cloud (5,000 portfolios)
plt.scatter(
    volatilities_array,
    returns_array,
    s=8,
    alpha=0.4,
    color="skyblue",
    label="5,000 Simulated Portfolios",
)

# 2. Highlight Equal-Weight Portfolio
# Volatility: ~0.015204, Expected Return: ~0.000794
plt.scatter(
    0.015204,
    0.000794,
    color="red",
    s=80,
    edgecolor="black",
    zorder=5,
    label="Equal-Weight (1/N)",
)

# 3. Highlight 100% JPM Portfolio
# Volatility: ~0.020094, Expected Return: ~0.000750
plt.scatter(
    0.020094,
    0.000750,
    color="orange",
    s=80,
    edgecolor="black",
    zorder=5,
    label="100% JPM",
)

# Formatting
plt.xlabel(r"Daily Volatility ($\sigma_p$, Decimal)")
plt.ylabel("Daily Expected Return ($E[R_p]$, Decimal)")
plt.title("Simulated Portfolio Risk-Return Profile (5,000 Portfolios)")
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# LESSON 6.4: IDENTIFYING MAX SHARPE & MIN VOLATILITY PORTFOLIOS
# -----------------------------------------------------------------------------
# Find indices using argmax and argmin
max_sharpe_idx = np.argmax(sharpes_array)
min_vol_idx = np.argmin(volatilities_array)

# Extract Max Sharpe details
max_sharpe_w = weights_array[max_sharpe_idx]
max_sharpe_ret = returns_array[max_sharpe_idx]
max_sharpe_vol = volatilities_array[max_sharpe_idx]
max_sharpe_sr = sharpes_array[max_sharpe_idx]

# Extract Min Volatility details
min_vol_w = weights_array[min_vol_idx]
min_vol_ret = returns_array[min_vol_idx]
min_vol_vol = volatilities_array[min_vol_idx]
min_vol_sr = sharpes_array[min_vol_idx]

print("=== MAX SHARPE PORTFOLIO ===")
print(f"Index:       {max_sharpe_idx}")
print(f"Weights:     JPM={max_sharpe_w[0]:.3f}, BAC={max_sharpe_w[1]:.3f}, PG={max_sharpe_w[2]:.3f}, MSFT={max_sharpe_w[3]:.3f}")
print(f"Return:      {max_sharpe_ret:.6f}")
print(f"Volatility:  {max_sharpe_vol:.6f}")
print(f"Sharpe:      {max_sharpe_sr:.6f}")
print(f"Check (ret/vol): {max_sharpe_ret / max_sharpe_vol:.6f}\n")

print("=== MIN VOLATILITY PORTFOLIO ===")
print(f"Index:       {min_vol_idx}")
print(f"Weights:     JPM={min_vol_w[0]:.3f}, BAC={min_vol_w[1]:.3f}, PG={min_vol_w[2]:.3f}, MSFT={min_vol_w[3]:.3f}")
print(f"Return:      {min_vol_ret:.6f}")
print(f"Volatility:  {min_vol_vol:.6f}")
print(f"Sharpe:      {min_vol_sr:.6f}")
print(f"Check (ret/vol): {min_vol_ret / min_vol_vol:.6f}\n")

# -----------------------------------------------------------------------------
# LESSON 6.5: DOMINATION COUNTS
# -----------------------------------------------------------------------------
# Compute Equal-Weight metrics dynamically
w_eq = np.array([0.25, 0.25, 0.25, 0.25])
eq_ret = portfolio_return(w_eq, mean_returns)
eq_vol = portfolio_volatility(w_eq, Sigma)

# Define metrics for the four target portfolios
targets = {
    "Max Sharpe": (max_sharpe_ret, max_sharpe_vol),
    "Min Volatility": (min_vol_ret, min_vol_vol),
    "Equal-Weight": (eq_ret, eq_vol),
    "100% JPM": (test_ret, test_vol),
}

print("=== DOMINATION COUNTS ===")
for name, (ret, vol) in targets.items():
  # Count portfolios with strictly higher return AND strictly lower volatility
  dominated_by = ((returns_array > ret) & (volatilities_array < vol)).sum()
  print(f"{name:<15} dominated by {dominated_by:>4} portfolios")
print()

scatter = plt.scatter(
    volatilities_array,
    returns_array,
    c=sharpes_array,
    cmap="viridis",
    s=12,
    alpha=0.6,
)

# Add colorbar for Sharpe ratio
cbar = plt.colorbar(scatter)
cbar.set_label("Sharpe Ratio ($r_f = 0$)")

# 2. Highlight Max Sharpe Portfolio (Index 2135)
plt.scatter(
    max_sharpe_vol,
    max_sharpe_ret,
    color="red",
    marker="*",
    s=200,
    edgecolor="black",
    zorder=10,
    label=f"Max Sharpe (SR = {max_sharpe_sr:.3f})",
)

# 3. Highlight Min Volatility Portfolio (Index 348)
plt.scatter(
    min_vol_vol,
    min_vol_ret,
    color="blue",
    marker="D",
    s=100,
    edgecolor="black",
    zorder=10,
    label=f"Min Volatility ($\sigma$ = {min_vol_vol:.4f})",
)

# 4. Highlight Equal-Weight Benchmark
plt.scatter(
    0.015204,
    0.000794,
    color="white",
    marker="o",
    s=80,
    edgecolor="black",
    zorder=9,
    label="Equal-Weight (1/N)",
)

# Formatting
plt.xlabel(r"Daily Volatility ($\sigma_p$, Decimal)")
plt.ylabel(r"Daily Expected Return ($E[R_p]$, Decimal)")
plt.title("5,000 Simulated Portfolios: Efficient Frontier & Key Highlights")
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# VISUALIZATION: EFFICIENT FRONTIER & SHARPE MAP
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# 1. Cloud colored by Sharpe Ratio
scatter = plt.scatter(
    volatilities_array,
    returns_array,
    c=sharpes_array,
    cmap="viridis",
    s=12,
    alpha=0.6,
)

cbar = plt.colorbar(scatter)
cbar.set_label("Sharpe Ratio ($r_f = 0$)")

# 2. Max Sharpe Portfolio (Red Star)
plt.scatter(
    max_sharpe_vol,
    max_sharpe_ret,
    color="red",
    marker="*",
    s=200,
    edgecolor="black",
    zorder=10,
    label=f"Max Sharpe (SR = {max_sharpe_sr:.3f})",
)

# 3. Min Volatility Portfolio (Blue Diamond)
plt.scatter(
    min_vol_vol,
    min_vol_ret,
    color="blue",
    marker="D",
    s=100,
    edgecolor="black",
    zorder=10,
    label=f"Min Volatility ($\sigma$ = {min_vol_vol:.4f})",
)

# 4. Equal-Weight Portfolio (White Circle)
plt.scatter(
    eq_vol,
    eq_ret,
    color="white",
    marker="o",
    s=80,
    edgecolor="black",
    zorder=9,
    label="Equal-Weight (1/N)",
)

# 5. 100% JPM Portfolio (Orange Square)
plt.scatter(
    test_vol,
    test_ret,
    color="orange",
    marker="s",
    s=80,
    edgecolor="black",
    zorder=9,
    label="100% JPM",
)

plt.xlabel(r"Daily Volatility ($\sigma_p$, Decimal)")
plt.ylabel(r"Daily Expected Return ($E[R_p]$, Decimal)")
plt.title("5,000 Simulated Portfolios: Efficient Frontier & Domination Analysis")
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# LESSON 6.6: EFFICIENT FRONTIER IDENTIFICATION
# -----------------------------------------------------------------------------
efficient_indices = []

# Loop over all 5,000 portfolios using i
for i in range(n_portfolios):
  ret_i = returns_array[i]
  vol_i = volatilities_array[i]

  # Count how many portfolios in the dataset dominate portfolio i
  n_dominators = ((returns_array > ret_i) & (volatilities_array < vol_i)).sum()

  # If no portfolio dominates i, it belongs to the efficient frontier set
  if n_dominators == 0:
    efficient_indices.append(i)

# Convert to array for clean indexing
efficient_indices = np.array(efficient_indices)

print("=== EFFICIENT FRONTIER ===")
print(f"Number of Efficient Portfolios Found: {len(efficient_indices)}")

# Sort efficient frontier points by volatility for a clean connected line
eff_vols = volatilities_array[efficient_indices]
eff_rets = returns_array[efficient_indices]
sort_order = np.argsort(eff_vols)

sorted_eff_vols = eff_vols[sort_order]
sorted_eff_rets = eff_rets[sort_order]

is_efficient = np.array([
    ((returns_array > returns_array[i]) & (volatilities_array < volatilities_array[i])).sum() == 0
    for i in range(n_portfolios)
])

# Extract efficient portfolio metrics using boolean indexing
eff_vols = volatilities_array[is_efficient]
eff_rets = returns_array[is_efficient]

print("=== EFFICIENT FRONTIER CHECK ===")
print(f"Number of Efficient Portfolios Found: {is_efficient.sum()}")

# Check in one line that both Max Sharpe and Min Volatility indices are included
print(f"Star & Diamond Included: {is_efficient[max_sharpe_idx] and is_efficient[min_vol_idx]}")

# -----------------------------------------------------------------------------
# LESSON 6.6 CLOSING CHECK: MAXIMUM RETURN BOUNDARY (100% MSFT)
# -----------------------------------------------------------------------------
w_msft = np.array([0.0, 0.0, 0.0, 1.0])

msft_ret = portfolio_return(w_msft, mean_returns)
msft_vol = portfolio_volatility(w_msft, Sigma)
msft_sr = sharpe_ratio(msft_ret, msft_vol, 0.0)

print("=== MAXIMUM RETURN BOUNDARY (100% MSFT) ===")
print(f"Weights:     {w_msft}")
print(f"Return:      {msft_ret:.6f}")
print(f"Volatility:  {msft_vol:.6f}")
print(f"Sharpe:      {msft_sr:.6f}\n")

# -----------------------------------------------------------------------------
# VISUALIZATION: 5,000 PORTFOLIOS & THE 55 EFFICIENT POINTS
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# 1. Cloud colored by Sharpe Ratio
scatter = plt.scatter(
    volatilities_array,
    returns_array,
    c=sharpes_array,
    cmap="viridis",
    s=12,
    alpha=0.4,
)
cbar = plt.colorbar(scatter)
cbar.set_label("Sharpe Ratio ($r_f = 0$)")

# Sort efficient points by volatility to draw a clean line
sort_order = np.argsort(eff_vols)
sorted_eff_vols = eff_vols[sort_order]
sorted_eff_rets = eff_rets[sort_order]

# 2. Draw line and scatter points for all 55 efficient portfolios
plt.plot(
    sorted_eff_vols,
    sorted_eff_rets,
    color="black",
    linestyle="--",
    linewidth=1.5,
    zorder=7,
)
plt.scatter(
    sorted_eff_vols,
    sorted_eff_rets,
    color="black",
    s=25,
    zorder=8,
    label=f"Efficient Frontier ({is_efficient.sum()} Points)",
)

# 3. Highlight Max Sharpe Portfolio (Red Star)
plt.scatter(
    max_sharpe_vol,
    max_sharpe_ret,
    color="red",
    marker="*",
    s=200,
    edgecolor="black",
    zorder=10,
    label=f"Max Sharpe (SR = {max_sharpe_sr:.3f})",
)

# 4. Highlight Min Volatility Portfolio (Blue Diamond)
plt.scatter(
    min_vol_vol,
    min_vol_ret,
    color="blue",
    marker="D",
    s=100,
    edgecolor="black",
    zorder=10,
    label=f"Min Volatility ($\sigma$ = {min_vol_vol:.4f})",
)

# 5. Highlight Equal-Weight & 100% JPM Benchmarks
plt.scatter(
    eq_vol,
    eq_ret,
    color="white",
    marker="o",
    s=80,
    edgecolor="black",
    zorder=9,
    label="Equal-Weight (1/N)",
)
plt.scatter(
    test_vol,
    test_ret,
    color="orange",
    marker="s",
    s=80,
    edgecolor="black",
    zorder=9,
    label="100% JPM",
)

plt.xlabel(r"Daily Volatility ($\sigma_p$, Decimal)")
plt.ylabel(r"Daily Expected Return ($E[R_p]$, Decimal)")
plt.title("5,000 Simulated Portfolios: 55 Empirically Derived Efficient Points")
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()