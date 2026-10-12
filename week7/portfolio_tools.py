from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

def _neg_sharpe(w, mean_returns, cov_matrix):
    """Objective function: negative Sharpe ratio for long-only portfolio."""
    port_return = w @ mean_returns
    port_vol = np.sqrt(w @ cov_matrix @ w)
    return -port_return / port_vol


def weights_sum_to_one(weights):
    """Constraint helper enforcing portfolio weights sum to 1.0."""
    return weights.sum() - 1.0


# --- Weight Selection Rules & Solvers ---

def unconstrained_weights(mean_array, cov_array):
    """Unconstrained analytical maximum-Sharpe portfolio weights."""
    positions = np.linalg.solve(cov_array, mean_array)
    return positions / positions.sum()


def exact_long_only_weights(mean_array, cov_array, show_verdicts=False):
    """Combinatorial exact long-only maximum-Sharpe portfolio weights."""
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


def slsqp_weights(mean_returns, cov_matrix):
    """Long-only maximum-Sharpe weights found by SLSQP, serving as an independent check on the exact solver."""
    n_assets = len(mean_returns)
    w0 = np.full(n_assets, 1.0 / n_assets)

    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        _neg_sharpe,
        w0,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"SLSQP optimization failed: {result.message}")

    return result.x


def rolling_weights(returns, window_length, first_test_position, weight_rule):
    """Weights for each test month, chosen only from the window_length months before it."""
    chosen = []
    test_months = []

    for position in range(first_test_position, len(returns)):
        window = returns.iloc[position - window_length : position]
        weights = weight_rule(window)
        chosen.append(weights)
        test_months.append(returns.index[position])

    return pd.DataFrame(chosen, index=test_months, columns=returns.columns)

def sharpe_gap_se(sharpe_p, sharpe_b, correlation, months):
    """Standard error of (Sharpe of p) minus (Sharpe of b). The answer is in the units of the Sharpe ratios you pass in."""
    variance_bracket = 2 * (1 - correlation) + 0.5 * (sharpe_p**2 + sharpe_b**2 - 2 * sharpe_p * sharpe_b * correlation**2)
    return np.sqrt(variance_bracket / months)

def smallest_detectable_gap(se, significance=0.05, target_power=0.80):
    return (norm.ppf(1 - significance / 2) + norm.ppf(target_power)) * se

def power_to_detect(gap, se, significance):
    z_crit = norm.ppf(1 - significance / 2)
    return norm.cdf(gap / se - z_crit) + norm.cdf(-gap / se - z_crit)


def chance_realised_gap_positive(gap, se):
    return norm.cdf(gap / se)


def months_needed_to_detect(gap, base_months, se, significance, target_power):
    sdg = smallest_detectable_gap(se, significance, target_power)
    return base_months * (sdg / gap) ** 2

def paired_bootstrap_sharpe_gaps(returns_a, returns_b, replications, rng):
    """Sharpe ratio of a minus Sharpe ratio of b, recomputed on resampled months (monthly if the returns are monthly).
    Each resample draws months with replacement and keeps the two returns of a drawn month together."""
    paired = np.column_stack([returns_a, returns_b])
    months = len(paired)
    gaps = np.empty(replications)
    for replication in range(replications):
        positions = rng.choice(months, size=months, replace=True)
        resample = paired[positions]
        sharpes = resample.mean(axis=0) / resample.std(axis=0, ddof=1)
        gaps[replication] = sharpes[0] - sharpes[1]
    return gaps

def portfolio_returns(weights, returns):
    """Monthly excess return of a portfolio: each month's weights times that same month's return."""
    return (weights * returns.loc[weights.index]).sum(axis=1)


def exact_rule(window):
    """Exact long-only maximum-Sharpe weights for one window (design-log items 2 and 6)."""
    # a) mean excess return per stock, as a NumPy array
    mean_returns = window.mean().to_numpy()
    
    # b) covariance matrix, as a NumPy array
    cov_matrix = window.cov().to_numpy()
    
    # c) call exact_long_only_weights with a) and b)
    weights = exact_long_only_weights(mean_returns, cov_matrix)
    
    # d) if it gives back None, stop with a clear error naming the window's last month
    if weights is None:
        last_month = window.index[-1]
        raise ValueError(f"No positive-mean return in window ending {last_month}")
        
    # e) return the weights
    return weights
