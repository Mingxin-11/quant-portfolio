# First out-of-sample run (Lesson 7.6): the locked design, run once.
# Basis: monthly excess returns (stock return minus TB3MS / 100 / 12), test months
# January 2018 - September 2026. Long-only exact maximum-Sharpe weights, re-estimated
# monthly on the previous M months, against 1/N. This is not the Week 6 basis.

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path

from portfolio_tools import (
    exact_rule,
    paired_bootstrap_sharpe_gaps,
    portfolio_returns,
    rolling_weights,
    sharpe_gap_se,
    smallest_detectable_gap,
)

EXCESS_FILE = "data/monthly_excess_returns_2008-01_to_2026-09.csv"
FIRST_TEST_MONTH = "2018-01"
EXPECTED_TEST_MONTHS = 105
WINDOW_LENGTHS = [60, 120]
MONTHS_PER_YEAR = 12
ANNUALISING_FACTOR = np.sqrt(MONTHS_PER_YEAR)
REPLICATIONS = 5000
SEED = 42
Z_95 = norm.ppf(0.975)

# From the lock of 12 October 2026, 13:32 SGT
LOCK_BAND = {"M = 60 gap": 0.5, "M = 120 gap": 0.5, "M = 120 minus M = 60": 0.3}
LOCK_SDG_RANGE = {
    "M = 60 gap": (0.36, 0.98),
    "M = 120 gap": (0.36, 0.98),
    "M = 120 minus M = 60": (0.36, 0.50),
}



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
    standard_error =sharpe_gap_se(sharpe_a, sharpe_b, correlation, months) * ANNUALISING_FACTOR 
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


# ---- Part A: your four functions on toy cases, before any real data is touched
toy_sharpe = sharpe_ratio(pd.Series([0.01, 0.03]))
toy_turnover = monthly_turnover(pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]))
toy_se = annual_formula_se(0.0, 0.0, 0.5, 100)
run_checks({
    "sharpe_ratio on [0.01, 0.03] is sqrt(2)":
        toy_sharpe is not None and np.isclose(toy_sharpe, np.sqrt(2)),
    "monthly_turnover on the toy weights is 1.0":
        toy_turnover is not None and np.isclose(toy_turnover, 1.0),
    "annual_formula_se(0, 0, 0.5, 100) is 0.1 * sqrt(12)":
        toy_se is not None and np.isclose(toy_se, 0.1 * ANNUALISING_FACTOR),
    "formula_is_wider: True for (-1, 1) against (-0.5, 0.5), False the other way round":
        formula_is_wider((-1.0, 1.0), (-0.5, 0.5)) and not formula_is_wider((-0.5, 0.5), (-1.0, 1.0)),
})

# ---- Part B: data, weights, returns
excess_returns = pd.read_csv(EXCESS_FILE, index_col=0, parse_dates=True).to_period("M")
first_test_position = excess_returns.index.get_loc(FIRST_TEST_MONTH)
test_months = excess_returns.index[first_test_position:]
number_of_stocks = excess_returns.shape[1]

weights_by_name = {}
for window_length in WINDOW_LENGTHS:
    weights_by_name[f"M = {window_length}"] = rolling_weights(
        excess_returns, window_length, first_test_position, exact_rule
    )
weights_by_name["1/N"] = pd.DataFrame(
    1 / number_of_stocks, index=test_months, columns=excess_returns.columns
)

returns_by_name = {}
for name, weights in weights_by_name.items():
    returns_by_name[name] = portfolio_returns(weights, excess_returns)

first_month = test_months[0]
by_hand = weights_by_name["M = 60"].loc[first_month].to_numpy() @ excess_returns.loc[first_month].to_numpy()
data_checks = {
    f"{EXPECTED_TEST_MONTHS} test months, 2018-01 to 2026-09":
        len(test_months) == EXPECTED_TEST_MONTHS
        and str(test_months[0]) == "2018-01"
        and str(test_months[-1]) == "2026-09",
    "M = 60, first test month: return equals weights @ returns by hand":
        np.isclose(returns_by_name["M = 60"].loc[first_month], by_hand),
}
for name, weights in weights_by_name.items():
    portfolio = returns_by_name[name]
    data_checks[f"{name}: weights cover the test months, none missing"] = (
        weights.index.equals(test_months) and not weights.isna().any().any()
    )
    data_checks[f"{name}: weights non-negative and add up to 1 every month"] = (
        weights.min().min() >= -1e-12 and np.allclose(weights.sum(axis=1), 1.0)
    )
    data_checks[f"{name}: one return per test month, none missing"] = (
        len(portfolio) == EXPECTED_TEST_MONTHS and not portfolio.isna().any()
    )
run_checks(data_checks)

# ---- Part C: results table, turnover, concentration
print("\nResults: monthly excess returns, 2018-01 to 2026-09, r_f = TB3MS, long-only, rebalanced monthly, no costs")
print(f"{'portfolio':<10}{'mean (ann.)':>13}{'vol (ann.)':>12}{'Sharpe (ann.)':>15}{'Sharpe (monthly)':>18}")
for name, portfolio in returns_by_name.items():
    monthly_sharpe = sharpe_ratio(portfolio)
    annual_mean = portfolio.mean() * MONTHS_PER_YEAR
    annual_volatility = portfolio.std() * ANNUALISING_FACTOR
    annual_sharpe = monthly_sharpe * ANNUALISING_FACTOR
    print(f"{name:<10}{annual_mean:>13.4f}{annual_volatility:>12.4f}{annual_sharpe:>15.4f}{monthly_sharpe:>18.4f}")

print("\nOptimiser trading and concentration (target weights month to month; drift inside the month ignored)")
for window_length in WINDOW_LENGTHS:
    weights = weights_by_name[f"M = {window_length}"]
    average_largest_weight = weights.max(axis=1).mean()
    print(
        f"M = {window_length}: average monthly turnover {monthly_turnover(weights):.3f} "
        f"(a full switch between two stocks = 2.0), average largest weight {average_largest_weight:.3f}"
    )

# ---- Part D: the three comparisons
comparisons = [
    ("M = 60 gap", "M = 60", "1/N"),
    ("M = 120 gap", "M = 120", "1/N"),
    ("M = 120 minus M = 60", "M = 120", "M = 60"),
]
summaries = {}
for label, first_name, second_name in comparisons:
    summaries[label] = gap_summary(returns_by_name[first_name], returns_by_name[second_name])
    print_gap(label, summaries[label], len(test_months))

# ---- Part E: the lock scored (True = the outcome the lock called more likely)
print("\nThe lock of 12 October 2026, 13:32 SGT, scored")
for label, summary in summaries.items():
    sdg = summary["smallest_detectable_gap"]
    low_sdg, high_sdg = LOCK_SDG_RANGE[label]
    print(label)
    show("(a) gap is positive", summary["gap"] > 0)
    show(f"(b) gap within +/-{LOCK_BAND[label]}", abs(summary["gap"]) <= LOCK_BAND[label])
    show("(c) zero inside wider interval", summary["zero_inside"])
    show(f"(d) SDG {sdg:.3f} inside {low_sdg}-{high_sdg}", low_sdg <= sdg and sdg <= high_sdg)

# ---- Part F: charts (added after the first run; Parts A-E are unchanged)
FIGURE_FOLDER = Path("figures/week7")
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)
BASIS_TEXT = "monthly excess returns, 2018-01 to 2026-09 (105 months), long-only, rebalanced monthly, no costs"


def finish_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(FIGURE_FOLDER / filename, dpi=200)
    plt.close(fig)
    print(f"saved {FIGURE_FOLDER / filename}")


def plot_gap_intervals(summaries, filename):
    labels = list(summaries.keys())
    gaps = []
    left_arms = []
    right_arms = []
    for label in labels:
        gap = summaries[label]["gap"]
        low, high = summaries[label]["bootstrap_interval"]
        gaps.append(gap)
        left_arms.append(gap - low)
        right_arms.append(high - gap)

    fig, ax = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(labels))
    ax.errorbar(gaps, positions, xerr=[left_arms, right_arms], fmt="o", capsize=5)
    ax.axvline(0, linestyle="--", color="grey")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Sharpe-ratio gap, annualised (first minus second)")
    title = (
        "Out-of-sample Sharpe-ratio gaps with 95% intervals\n"
        f"{BASIS_TEXT}\n"
        "bars: paired-bootstrap percentile intervals, 5,000 resamples, seed 42"
    )
    ax.set_title(title, fontsize=9)
    finish_figure(fig, filename)


def plot_weights(ax, weights, title):
    dates = weights.index.to_timestamp()
    ax.stackplot(dates, weights.to_numpy().T, labels=list(weights.columns))
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("weight")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", ncol=4, fontsize=8)


def plot_all_weights(weights_by_name, filename):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6.5))
    plot_weights(axes[0], weights_by_name["M = 60"], "M = 60: optimiser weights by test month")
    plot_weights(axes[1], weights_by_name["M = 120"], "M = 120: optimiser weights by test month")
    axes[1].set_xlabel(f"test month; {BASIS_TEXT}", fontsize=8)
    finish_figure(fig, filename)


def plot_growth(returns_by_name, filename):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, portfolio in returns_by_name.items():
        growth = (1 + portfolio).cumprod()
        ax.plot(growth.index.to_timestamp(), growth, label=name)
    ax.axhline(1.0, color="grey", linewidth=0.8)
    ax.set_xlabel("month")
    ax.set_ylabel("growth of 1 unit of compounded monthly excess returns")
    ax.set_title(f"Compounded excess returns: one realised path\n{BASIS_TEXT}", fontsize=9)
    ax.legend()
    finish_figure(fig, filename)


plot_gap_intervals(summaries, "first_run_gap_intervals.png")
plot_all_weights(weights_by_name, "first_run_weights.png")
plot_growth(returns_by_name, "first_run_growth.png")