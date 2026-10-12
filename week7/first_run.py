# First out-of-sample run (Lesson 7.6): the locked design, run once.
# Basis: monthly excess returns (stock return minus TB3MS / 100 / 12), test months
# January 2018 - September 2026. Long-only exact maximum-Sharpe weights, re-estimated
# monthly on the previous M months, against 1/N. This is not the Week 6 basis.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from portfolio_tools import (
    ANNUALISING_FACTOR,
    EXCESS_FILE,
    EXPECTED_TEST_MONTHS,
    FIRST_TEST_MONTH,
    MONTHS_PER_YEAR,
    WINDOW_LENGTHS,
    annual_formula_se,
    build_strategies,
    formula_is_wider,
    gap_summary,
    load_excess_returns,
    monthly_turnover,
    print_gap,
    run_checks,
    sharpe_ratio,
    show,
)

# From the lock of 12 October 2026, 13:32 SGT
LOCK_BAND = {"M = 60 gap": 0.5, "M = 120 gap": 0.5, "M = 120 minus M = 60": 0.3}
LOCK_SDG_RANGE = {
    "M = 60 gap": (0.36, 0.98),
    "M = 120 gap": (0.36, 0.98),
    "M = 120 minus M = 60": (0.36, 0.50),
}

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
excess_returns = load_excess_returns(EXCESS_FILE)
first_test_position = excess_returns.index.get_loc(FIRST_TEST_MONTH)
test_months = excess_returns.index[first_test_position:]
weights_by_name, returns_by_name = build_strategies(excess_returns, FIRST_TEST_MONTH, WINDOW_LENGTHS)

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