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


# --- Weight Selection Rules & Solvers ---

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
    mean_returns = window.mean().to_numpy()   # average monthly excess return of each stock
    cov_matrix = window.cov().to_numpy()      # covariance of the stocks' monthly excess returns

    weights = exact_long_only_weights(mean_returns, cov_matrix)

    # None means the solver found no single solution, for example when no stock has a positive
    # mean in the window (design-log item 6): stop with an error that names the window.
    if weights is None:
        last_month = window.index[-1]
        raise ValueError(f"No positive-mean return in window ending {last_month}")

    return weights

# --- Study settings and helpers (moved from first_run.py in Lesson 8.1, Step 4) ---
EXCESS_FILE = "data/monthly_excess_returns_2008-01_to_2026-09.csv"
FIRST_TEST_MONTH = "2018-01"
EXPECTED_TEST_MONTHS = 105
WINDOW_LENGTHS = [60, 120]
MONTHS_PER_YEAR = 12
ANNUALISING_FACTOR = np.sqrt(MONTHS_PER_YEAR)
REPLICATIONS = 5000
SEED = 42
Z_95 = norm.ppf(0.975)


def sharpe_ratio(returns):
    # Monthly Sharpe ratio of a Series of monthly excess returns
    monthly_sharpe = returns.mean() / returns.std()
    return monthly_sharpe


def monthly_turnover(weights):
    # Average over months of the sum of absolute weight changes since the previous month
    average_turnover = weights.diff().dropna().abs().sum(axis=1).mean()
    return average_turnover


def annual_formula_se(sharpe_a, sharpe_b, correlation, months):
    # Annualised formula standard error of the gap between two MONTHLY Sharpe ratios
    standard_error = sharpe_gap_se(sharpe_a, sharpe_b, correlation, months) * ANNUALISING_FACTOR
    return standard_error


def formula_is_wider(formula_interval, bootstrap_interval):
    # True when the formula interval is wider than the bootstrap interval
    is_wider = (formula_interval[1] - formula_interval[0]) > (bootstrap_interval[1] - bootstrap_interval[0])
    return is_wider


# ---- Helpers: checks, one comparison, printing

def run_checks(checks):
    for description, passed in checks.items():
        print(f"{passed}  {description}")
    if not all(checks.values()):
        raise RuntimeError("A check is False: stopping before any result.")


def gap_summary(returns_a, returns_b):
    # Sharpe gap (a minus b), annualised, with the formula and the bootstrap interval
    months = len(returns_a)
    sharpe_a = sharpe_ratio(returns_a)
    sharpe_b = sharpe_ratio(returns_b)
    correlation = returns_a.corr(returns_b)
    gap = (sharpe_a - sharpe_b) * ANNUALISING_FACTOR

    formula_se = annual_formula_se(sharpe_a, sharpe_b, correlation, months)
    formula_interval = (gap - Z_95 * formula_se, gap + Z_95 * formula_se)

    rng = np.random.default_rng(SEED)
    monthly_gaps = paired_bootstrap_sharpe_gaps(returns_a, returns_b, REPLICATIONS, rng)
    bootstrap_gaps = monthly_gaps * ANNUALISING_FACTOR
    bootstrap_se = bootstrap_gaps.std(ddof=1)
    bootstrap_interval = tuple(np.percentile(bootstrap_gaps, [2.5, 97.5]))

    formula_wins = formula_is_wider(formula_interval, bootstrap_interval)
    quoted_interval = formula_interval if formula_wins else bootstrap_interval
    larger_se = max(formula_se, bootstrap_se)
    return {
        "gap": gap,
        "correlation": correlation,
        "formula_se": formula_se,
        "formula_interval": formula_interval,
        "bootstrap_se": bootstrap_se,
        "bootstrap_interval": bootstrap_interval,
        "formula_is_wider": formula_wins,
        "quoted_interval": quoted_interval,
        "zero_inside": quoted_interval[0] <= 0 and 0 <= quoted_interval[1],
        "smallest_detectable_gap": smallest_detectable_gap(larger_se),
    }


def annual_and_monthly(value):
    return f"{value:.3f}  (monthly {value / ANNUALISING_FACTOR:.4f})"


def interval_text(interval):
    low, high = interval
    monthly_low = low / ANNUALISING_FACTOR
    monthly_high = high / ANNUALISING_FACTOR
    return f"[{low:+.3f}, {high:+.3f}]  (monthly [{monthly_low:+.4f}, {monthly_high:+.4f}])"


def show(label, text):
    print(f"  {label:<34}{text}")


def print_gap(label, summary, months):
    gap = summary["gap"]
    print(f"\n{label} ({months} test months)")
    show("gap", f"{gap:+.3f}  (monthly {gap / ANNUALISING_FACTOR:+.4f})")
    show("correlation of the two series", f"{summary['correlation']:.3f}")
    show("formula SE", annual_and_monthly(summary["formula_se"]))
    show("formula interval", interval_text(summary["formula_interval"]))
    show("bootstrap SE", annual_and_monthly(summary["bootstrap_se"]))
    show("bootstrap interval", interval_text(summary["bootstrap_interval"]))
    show("bootstrap SE / formula SE", f"{summary['bootstrap_se'] / summary['formula_se']:.3f}")
    wider_name = "formula" if summary["formula_is_wider"] else "bootstrap"
    show("wider interval", f"{wider_name}: {interval_text(summary['quoted_interval'])}")
    show("zero inside wider interval", summary["zero_inside"])
    show("smallest detectable gap", f"{summary['smallest_detectable_gap']:.3f}  (from the larger SE)")

def load_excess_returns(filename):
    """Monthly excess returns, one column per stock, indexed by month."""
    return pd.read_csv(filename, index_col=0, parse_dates=True).to_period("M")


def build_strategies(excess_returns, first_test_month, window_lengths):
    """Weights and returns of each strategy (optimiser per window length, and 1/N) over the test months."""
    first_test_position = excess_returns.index.get_loc(first_test_month)
    test_months = excess_returns.index[first_test_position:]
    number_of_stocks = excess_returns.shape[1]

    weights_by_name = {}
    for window_length in window_lengths:
        weights_by_name[f"M = {window_length}"] = rolling_weights(
            excess_returns, window_length, first_test_position, exact_rule
        )
    weights_by_name["1/N"] = pd.DataFrame(
        1 / number_of_stocks, index=test_months, columns=excess_returns.columns
    )

    returns_by_name = {}
    for name, weights in weights_by_name.items():
        returns_by_name[name] = portfolio_returns(weights, excess_returns)
    return weights_by_name, returns_by_name



