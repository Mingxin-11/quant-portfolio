import numpy as np
import pandas as pd

from portfolio_tools import paired_bootstrap_sharpe_gaps, sharpe_gap_se

REPLICATIONS = 5000                # resamples for each bootstrap on real data
CHECK_REPLICATIONS = 1000          # resamples per bootstrap inside the calibration check, which repeats it SAMPLES times
SAMPLES = 200                      # fresh simulated samples per world in the calibration check
TEST_MONTHS = 105                  # January 2018 to September 2026
SEED = 42
ANNUALISING_FACTOR = np.sqrt(12)   # monthly Sharpe ratio, or its SE, to a yearly figure
CALIBRATION_TOLERANCE = 0.05       # "calibrated" means the average bootstrap SE is within 5% of the formula SE
LOWER_PERCENT = 2.5                # a 95% interval leaves 2.5% in each tail
UPPER_PERCENT = 97.5

# (name, true monthly Sharpe of a, true monthly Sharpe of b, true correlation): the two Step 3 worlds
worlds = [
    ("A  JPM-like", 0.150, 0.139, 0.885),
    ("B  PG-like", 0.118, 0.139, 0.471),
]


def average_bootstrap_se(sharpe_a, sharpe_b, correlation, months, samples, replications, rng):
    true_means = [sharpe_a, sharpe_b]
    true_cov = [[1.0, correlation], [correlation, 1.0]]
    standard_errors = []
    for _ in range(samples):
        sample = rng.multivariate_normal(true_means, true_cov, size=months)
        gaps = paired_bootstrap_sharpe_gaps(sample[:, 0], sample[:, 1], replications, rng)
        standard_errors.append(gaps.std(ddof=1))
    return np.mean(standard_errors)


rng = np.random.default_rng(SEED)

print("Part A: the pairing check")
identical_months = rng.normal(loc=0.01, scale=0.05, size=TEST_MONTHS)
identical_gaps = paired_bootstrap_sharpe_gaps(identical_months, identical_months, REPLICATIONS, rng)
print("   two identical series give a gap of exactly zero in every resample:", np.allclose(identical_gaps, 0.0, atol=1e-12))

print(f"Part B: in a world of independent normal months the bootstrap SE should match the formula ({SAMPLES} samples of {TEST_MONTHS} months per world; monthly units)")
print("   world        | formula SE | average bootstrap SE | bootstrap / formula | within tolerance")
for name, sharpe_a, sharpe_b, correlation in worlds:
    formula_se = sharpe_gap_se(sharpe_a, sharpe_b, correlation, TEST_MONTHS)
    bootstrap_se = average_bootstrap_se(sharpe_a, sharpe_b, correlation, TEST_MONTHS, SAMPLES, CHECK_REPLICATIONS, rng)
    ratio = bootstrap_se / formula_se
    print(f"   {name:12} | {formula_se:10.5f} | {bootstrap_se:20.5f} | {ratio:19.4f} | {abs(ratio - 1) < CALIBRATION_TOLERANCE}")

excess_returns = pd.read_csv(
    "data/monthly_excess_returns_2008-01_to_2026-09.csv",
    index_col=0,
    parse_dates=True,
).to_period("M")

planning_returns = excess_returns.loc["2008-01":"2017-12"]
planning_months = len(planning_returns)
equal_weight_returns = planning_returns.mean(axis=1)
equal_sharpe = equal_weight_returns.mean() / equal_weight_returns.std()

print(f"Part C: each stock against 1/N on the {planning_months} planning months (annualised): is the formula too narrow on real data?")
print("   anchor | formula SE | bootstrap SE | bootstrap / formula | bootstrap 95% interval for the planning gap")
for ticker in planning_returns.columns:
    stock_returns = planning_returns[ticker]
    stock_sharpe = stock_returns.mean() / stock_returns.std()
    correlation = stock_returns.corr(equal_weight_returns)
    formula_se = sharpe_gap_se(stock_sharpe, equal_sharpe, correlation, planning_months) * ANNUALISING_FACTOR
    gaps = paired_bootstrap_sharpe_gaps(stock_returns, equal_weight_returns, REPLICATIONS, rng) * ANNUALISING_FACTOR
    bootstrap_se = gaps.std(ddof=1)
    lower, upper = np.percentile(gaps, [LOWER_PERCENT, UPPER_PERCENT])
    print(f"   {ticker:6} | {formula_se:10.3f} | {bootstrap_se:12.3f} | {bootstrap_se / formula_se:19.3f} | {lower:+.3f} to {upper:+.3f}")