import numpy as np

from portfolio_tools import sharpe_gap_se

REPLICATIONS = 80000
MONTHS = 105
SEED = 42
MONTE_CARLO_ERROR = 1 / np.sqrt(2 * (REPLICATIONS - 1))
FINITE_SAMPLE_ALLOWANCE = 0.02
RATIO_TOLERANCE = 3 * MONTE_CARLO_ERROR + FINITE_SAMPLE_ALLOWANCE

# (name, true monthly Sharpe of p, true monthly Sharpe of b, true correlation)
worlds = [
    ("A  JPM-like", 0.150, 0.139, 0.885),
    ("B  PG-like", 0.118, 0.139, 0.471),
    ("C  rho 0, large", 1.0, 1.0, 0.0),
    ("D  rho 0.9, large", 1.0, 0.5, 0.9),
]


def simulated_gap_sd(sharpe_p, sharpe_b, correlation, months, replications, rng):
    true_means = [sharpe_p, sharpe_b]
    true_cov = [[1.0, correlation], [correlation, 1.0]]
    gaps = []
    for _ in range(replications):
        sample = rng.multivariate_normal(true_means, true_cov, size=months)
        sample_sharpes = sample.mean(axis=0) / sample.std(axis=0, ddof=1)
        gaps.append(sample_sharpes[0] - sample_sharpes[1])
    return np.std(gaps, ddof=1)


rng = np.random.default_rng(SEED)
print(f"Monte Carlo error of a simulated sd with {REPLICATIONS} replications: {MONTE_CARLO_ERROR:.4f}; pass line: 3 of them plus {FINITE_SAMPLE_ALLOWANCE} = {RATIO_TOLERANCE:.4f}")
print("   world              | formula (full) | first term only | simulated sd | simulated / formula | full formula agrees | first term alone agrees")
for name, sharpe_p, sharpe_b, correlation in worlds:
    formula = sharpe_gap_se(sharpe_p, sharpe_b, correlation, MONTHS)
    first_term_only = np.sqrt(2 * (1 - correlation) / MONTHS)
    simulated = simulated_gap_sd(sharpe_p, sharpe_b, correlation, MONTHS, REPLICATIONS, rng)
    full_agrees = abs((simulated / formula) - 1.0) < RATIO_TOLERANCE
    first_term_agrees = abs((simulated / first_term_only) - 1.0) < RATIO_TOLERANCE
    print(f"   {name:18} | {formula:14.5f} | {first_term_only:15.5f} | {simulated:12.5f} | {simulated / formula:19.4f} | {str(full_agrees):19} | {first_term_agrees}")