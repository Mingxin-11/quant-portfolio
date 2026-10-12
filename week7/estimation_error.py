import numpy as np
import pandas as pd

from portfolio_tools import exact_long_only_weights

REPLICATIONS = 5000
WINDOW_LENGTHS = [60, 120]
SEED = 42
ANNUALISING_FACTOR = np.sqrt(12)

excess_returns = pd.read_csv(
    "data/monthly_excess_returns_2008-01_to_2026-09.csv",
    index_col=0,
    parse_dates=True,
).to_period("M")

planning_returns = excess_returns.loc["2008-01":"2017-12"]
planning_means = planning_returns.mean().to_numpy()
planning_cov = planning_returns.cov().to_numpy()
planning_sds = np.sqrt(np.diag(planning_cov))
equal_weights = np.full(4, 0.25)

# (name, true means, true covariance matrix)
truths = [
    ("1  planning data as truth", planning_means, planning_cov),
    ("2  equal Sharpe ratios", planning_sds * (planning_means / planning_sds).mean(), planning_cov),
]


def true_sharpe(weights, true_means, true_cov):
    # YOUR LINE 1: return the Sharpe ratio of these weights under these means and this covariance matrix (Week 5's w'mu and w'Sigma w)
    return weights @ true_means / np.sqrt(weights @ true_cov @ weights)


def simulate_estimated_optimiser(true_means, true_cov, window_length, replications, rng):
    true_sharpes = []
    in_sample_sharpes = []
    largest_weights = []
    fallbacks = 0
    for _ in range(replications):
        # YOUR LINE 2: sample = window_length months of returns drawn from the truth (as in Step 3, but four stocks)
        sample = rng.multivariate_normal(true_means, true_cov, size=window_length)
        sample_means = sample.mean(axis=0)
        sample_cov = np.cov(sample, rowvar=False)
        estimated_weights = exact_long_only_weights(sample_means, sample_cov)
        if estimated_weights is None:
            estimated_weights = equal_weights
            fallbacks += 1
        # YOUR LINE 3: add the TRUE Sharpe ratio of estimated_weights to true_sharpes
        true_sharpes.append(true_sharpe(estimated_weights, true_means, true_cov))
        # YOUR LINE 4: add the IN-SAMPLE Sharpe ratio of estimated_weights (judged by the sample's own means and covariance) to in_sample_sharpes
        in_sample_sharpes.append(true_sharpe(estimated_weights, sample_means, sample_cov))
        largest_weights.append(estimated_weights.max())
    return np.array(true_sharpes), np.array(in_sample_sharpes), np.array(largest_weights), fallbacks


rng = np.random.default_rng(SEED)
print("Planning months:", len(planning_returns), "| first:", planning_returns.index[0], "| last:", planning_returns.index[-1])
print(f"Sharpe ratios below are annualised (x sqrt 12) and monthly-based; {REPLICATIONS} replications; seed {SEED}")
for name, true_means, true_cov in truths:
    equal_sharpe = true_sharpe(equal_weights, true_means, true_cov)
    optimum_sharpe = true_sharpe(exact_long_only_weights(true_means, true_cov), true_means, true_cov)
    print(f"Truth {name}: true Sharpe of 1/N {ANNUALISING_FACTOR * equal_sharpe:.3f} | true Sharpe of the best long-only portfolio {ANNUALISING_FACTOR * optimum_sharpe:.3f}")
    print("   window | in-sample Sharpe of the estimated weights | true Sharpe of the estimated weights | share beating 1/N | mean largest weight | fallbacks to 1/N | none above the optimum")
    for window_length in WINDOW_LENGTHS:
        true_sharpes, in_sample_sharpes, largest_weights, fallbacks = simulate_estimated_optimiser(true_means, true_cov, window_length, REPLICATIONS, rng)
        share_beating = np.mean(true_sharpes > equal_sharpe)
        none_above = np.all(true_sharpes <= optimum_sharpe + 1e-9)
        print(f"   {window_length:6} | {ANNUALISING_FACTOR * in_sample_sharpes.mean():41.3f} | {ANNUALISING_FACTOR * true_sharpes.mean():36.3f} | {share_beating:17.3f} | {largest_weights.mean():19.3f} | {fallbacks:16} | {none_above}")