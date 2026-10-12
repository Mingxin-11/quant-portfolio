import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import minimize

# Load data and setup metrics
prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
returns = prices.pct_change().dropna()

# Enforce stock order: JPM, BAC, PG, MSFT
returns = returns[["JPM", "BAC", "PG", "MSFT"]]

mean_returns = returns.mean()
cov_matrix = returns.cov()

# Data Check
print("Data check (mean_returns):")
print(mean_returns.round(6))
print("-" * 50)

risk_aversion = 1.0

mean_array = mean_returns.to_numpy()
cov_array = cov_matrix.to_numpy()

positions = np.linalg.solve(cov_array, mean_array) / risk_aversion
total_position = positions.sum()
relative_weights = positions / total_position

portfolio_mean = relative_weights @ mean_array
portfolio_std = np.sqrt(relative_weights @ cov_array @ relative_weights)

print("positions:", positions)
print("total position:", total_position)
print("relative weights:", relative_weights)
print("daily Sharpe (r_f = 0):", portfolio_mean / portfolio_std)
print("solves the equation:", np.allclose(risk_aversion * cov_array @ positions, mean_array))

free_assets = [0,2,3]

free_cov = cov_array[np.ix_(free_assets, free_assets)]
free_mean = mean_array[free_assets]

constrained_positions = np.zeros(4)
constrained_positions[free_assets] = np.linalg.solve(free_cov, free_mean) / risk_aversion

multipliers = risk_aversion * cov_array @ constrained_positions - mean_array
constrained_weights = constrained_positions / constrained_positions.sum()
adjusted_mean = mean_array + multipliers
adjusted_positions = np.linalg.solve(cov_array, adjusted_mean) / risk_aversion

constrained_mean = constrained_weights @ mean_array
constrained_std = np.sqrt(constrained_weights @ cov_array @ constrained_weights)

print("constrained positions:", constrained_positions)
print("multipliers (lambda):", multipliers)
print("constrained relative weights:", constrained_weights)
print("constrained daily Sharpe (r_f = 0):", constrained_mean / constrained_std)
print("adjusted mean (mu + lambda):", adjusted_mean)
print("adjusted-mean solve matches constrained positions:", np.allclose(adjusted_positions, constrained_positions))

def daily_sharpe(weights, mean_array, cov_array):
    portfolio_mean = weights @ mean_array
    portfolio_std = np.sqrt(weights @ cov_array @ weights)
    return portfolio_mean / portfolio_std

def negative_sharpe(weights, mean_array, cov_array):
    return -daily_sharpe(weights, mean_array, cov_array)


def weights_sum_to_one(weights):
    return weights.sum() - 1


def optimise_sharpe(starting_weights, mean_array, cov_array, ftol=1e-6):
    return minimize(
        negative_sharpe,
        starting_weights,
        args=(mean_array, cov_array),
        method="SLSQP",
        bounds=[(0, None)] * len(starting_weights),
        constraints={"type": "eq", "fun": weights_sum_to_one},
        options={"ftol": ftol},
    )


equal_start = np.full(4, 0.25)
skewed_start = np.array([0.7, 0.1, 0.1, 0.1])

runs = [
    ("default tolerance, equal start", optimise_sharpe(equal_start, mean_array, cov_array)),
    ("default tolerance, skewed start", optimise_sharpe(skewed_start, mean_array, cov_array)),
    ("tight tolerance, equal start", optimise_sharpe(equal_start, mean_array, cov_array, ftol=1e-12)),
]

for label, result in runs:
    print(label)
    print("  weights:", result.x)
    print("  daily Sharpe (r_f = 0):", -result.fun)
    print("  converged:", result.success)
    print("  largest weight difference from exact:", np.max(np.abs(result.x - constrained_weights)))

def unconstrained_weights(mean_array, cov_array):
    positions = np.linalg.solve(cov_array, mean_array)
    return positions / positions.sum()


mean_standard_errors = returns.std() / np.sqrt(len(returns))
print("standard errors of the daily means:")
print(mean_standard_errors)
print("each mean measured in standard errors:")
print(mean_returns / mean_standard_errors)

bac_index = 1
bac_mean_standard_error = mean_standard_errors["BAC"]
shifts = [-0.0004, -0.0002, 0.0, 0.0001, 0.0002, 0.0002273, 0.0003, 0.0004, 0.0006]

original_unconstrained = unconstrained_weights(mean_array, cov_array)
original_long_only = optimise_sharpe(equal_start, mean_array, cov_array, ftol=1e-12).x
best_unconstrained_sharpe = daily_sharpe(original_unconstrained, mean_array, cov_array)
best_long_only_sharpe = daily_sharpe(original_long_only, mean_array, cov_array)

for shift in shifts:
    shifted_mean = mean_array.copy()
    shifted_mean[bac_index] += shift

    unconstrained = unconstrained_weights(shifted_mean, cov_array)
    long_only = optimise_sharpe(equal_start, shifted_mean, cov_array, ftol=1e-12).x

    unconstrained_loss = 100 * (1 - daily_sharpe(unconstrained, mean_array, cov_array) / best_unconstrained_sharpe)
    long_only_loss = 100 * (1 - daily_sharpe(long_only, mean_array, cov_array) / best_long_only_sharpe)

    print("shift", shift, "=", round(shift / bac_mean_standard_error, 2), "standard errors")
    print("  unconstrained:", np.round(unconstrained, 3), "| Sharpe lost (%):", round(unconstrained_loss, 1))
    print("  long-only:    ", np.round(long_only, 3), "| Sharpe lost (%):", round(long_only_loss, 1))

candidate_totals = np.linspace(0, 10, 1001)

direction_mean = constrained_weights @ mean_array
direction_variance = constrained_weights @ cov_array @ constrained_weights

utilities = candidate_totals * direction_mean - (risk_aversion / 2) * candidate_totals ** 2 * direction_variance

best_index = utilities.argmax()
print("scan: best total position:", candidate_totals[best_index])
print("scan: best utility:", utilities[best_index])
print("exact long-only positions add up to:", constrained_positions.sum())


formula_best_t = direction_mean / (risk_aversion * direction_variance)
formula_best_u = (direction_mean**2) / (2 * risk_aversion * direction_variance)

print("formula: best total position (t*):", formula_best_t)
print("formula: best utility U(t*):", formula_best_u)

def exact_long_only_weights(mean_array, cov_array, show_verdicts=False):
    asset_count = len(mean_array)
    tolerance = 1e-12
    passing_weights = []

    for subset_size in range(1, asset_count + 1):
        for free_tuple in combinations(range(asset_count), subset_size):
            free_assets = list(free_tuple)

            positions = np.zeros(asset_count)
            positions[free_assets] = np.linalg.solve(
                cov_array[np.ix_(free_assets, free_assets)],
                mean_array[free_assets],
            )
            multipliers = cov_array @ positions - mean_array

            free_positions_positive = np.all(
                positions[free_assets] > tolerance
            )
            pinned_multipliers_ok = np.all(
                np.delete(multipliers, free_assets) >= -tolerance
            )

            if show_verdicts:
                print(
                    "free assets",
                    free_assets,
                    "| free positions > 0:",
                    free_positions_positive,
                    "| pinned multipliers >= 0:",
                    pinned_multipliers_ok,
                )

            if free_positions_positive and pinned_multipliers_ok:
                passing_weights.append(positions / positions.sum())

    if len(passing_weights) == 1:
        return passing_weights[0]
    return None


# 2. Place the execution calls at the very bottom of the file:
exact_weights = exact_long_only_weights(mean_array, cov_array)
slsqp_weights = optimise_sharpe(
    equal_start, mean_array, cov_array, ftol=1e-12
).x

print("exact solver weights:", exact_weights)
print(
    "same as your 3a weights:", np.allclose(exact_weights, constrained_weights)
)
print(
    "largest difference from SLSQP (ftol 1e-12):",
    np.max(np.abs(exact_weights - slsqp_weights)),
)
print("all means negative ->", exact_long_only_weights(-mean_array, cov_array))

exact_long_only_weights(mean_array, cov_array, show_verdicts=True)

# Clearly named 1/N benchmark vector
one_over_n_weights = np.full(len(mean_array), 1.0 / len(mean_array))
one_over_n_sharpe = daily_sharpe(one_over_n_weights, mean_array, cov_array)
print("1/N daily Sharpe baseline:", round(one_over_n_sharpe, 6))

for shift in shifts:
    shifted_mean = mean_array.copy()
    shifted_mean[bac_index] += shift

    unconstrained = unconstrained_weights(shifted_mean, cov_array)
    long_only = exact_long_only_weights(shifted_mean, cov_array)

    # Scored under ORIGINAL true mean_array
    unconstrained_sharpe = daily_sharpe(
        unconstrained, mean_array, cov_array
    )
    long_only_sharpe = daily_sharpe(long_only, mean_array, cov_array)

    print(
        "shift",
        shift,
        "=",
        round(shift / bac_mean_standard_error, 2),
        "standard errors",
    )
    print(
        "  True Sharpes -> Unconstrained:",
        round(unconstrained_sharpe, 5),
        "| Long-only:",
        round(long_only_sharpe, 5),
        "| 1/N:",
        round(one_over_n_sharpe, 5),
    )