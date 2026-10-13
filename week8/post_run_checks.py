# Lesson 8.2: post-run checks on the first run of 12 October 2026.
# Everything printed under POST-RUN is reported whatever it shows; none of it replaces the locked result.
# Predictions locked 13 October 2026, 08:23 SGT (project doc claude/18_WEEK8_LOCK_RECORD.md).

import sys
from pathlib import Path

# week8 has no copy of portfolio_tools.py: add the week7 folder to the places Python looks for modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "week7"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio_tools import (
    ANNUALISING_FACTOR,
    EXCESS_FILE,
    EXPECTED_TEST_MONTHS,
    FIRST_TEST_MONTH,
    MONTHS_PER_YEAR,
    REPLICATIONS,
    SEED,
    WINDOW_LENGTHS,
    Z_95,
    build_strategies,
    formula_is_wider,
    gap_summary,
    load_excess_returns,
    monthly_turnover,
    paired_bootstrap_sharpe_gaps,
    run_checks,
    sharpe_ratio,
    smallest_detectable_gap,
)

SPLIT_MONTH = "2024-01"  # first month of sub-period B: the only cut, fixed before the run
FIRST_RUN_SHARPE = {"M = 60": 0.7845, "M = 120": 0.8855, "1/N": 0.7790}  # annualised, first-run printout

comparisons = [
    ("M = 60 gap", "M = 60", "1/N"),
    ("M = 120 gap", "M = 120", "1/N"),
    ("M = 120 minus M = 60", "M = 120", "M = 60"),
]


def split_periods(series, split_position):
    return series.iloc[:split_position], series.iloc[split_position:]


def difference_of_gaps(first_a, second_a, first_b, second_b, summary_a, summary_b):
    # (gap in B) minus (gap in A), treating the two periods as independent.
    # Approximate: the weights used in B were estimated on windows that reach back into A.
    difference = summary_b["gap"] - summary_a["gap"]

    # formula route: the variances of two independent estimates add
    formula_se = np.sqrt(summary_a["formula_se"] ** 2 + summary_b["formula_se"] ** 2)
    formula_interval = (difference - Z_95 * formula_se, difference + Z_95 * formula_se)

    # bootstrap route: resample A, then resample B (the second draw continues the same random
    # stream, so the two are independent), and subtract
    rng = np.random.default_rng(SEED)
    gaps_a = paired_bootstrap_sharpe_gaps(first_a, second_a, REPLICATIONS, rng) * ANNUALISING_FACTOR
    gaps_b = paired_bootstrap_sharpe_gaps(first_b, second_b, REPLICATIONS, rng) * ANNUALISING_FACTOR
    differences = gaps_b - gaps_a
    bootstrap_se = differences.std(ddof=1)
    bootstrap_interval = tuple(np.percentile(differences, [2.5, 97.5]))

    if formula_is_wider(formula_interval, bootstrap_interval):
        quoted_interval = formula_interval
    else:
        quoted_interval = bootstrap_interval
    return {
        "gap": difference,
        "formula_se": formula_se,
        "bootstrap_se": bootstrap_se,
        "quoted_interval": quoted_interval,
        "zero_inside": quoted_interval[0] <= 0 and 0 <= quoted_interval[1],
        "smallest_detectable_gap": smallest_detectable_gap(max(formula_se, bootstrap_se)),
    }


printed_rows = []  # every row print_row prints, kept for the figure in Part F


def print_row(label, period, summary):
    printed_rows.append((label, period, summary))
    low, high = summary["quoted_interval"]
    interval = f"[{low:+.3f}, {high:+.3f}]"
    print(
        f"{label:<22}{period:<7}{summary['gap']:>+8.3f}{summary['formula_se']:>12.3f}"
        f"{summary['bootstrap_se']:>10.3f}{summary['smallest_detectable_gap']:>8.3f}"
        f"   {interval:<20}{summary['zero_inside']}"
    )


# ---- Part A: rebuild the first run and check that it reproduces before anything new is computed
excess_returns = load_excess_returns(EXCESS_FILE)
weights_by_name, returns_by_name = build_strategies(excess_returns, FIRST_TEST_MONTH, WINDOW_LENGTHS)
test_months = returns_by_name["1/N"].index
split_position = test_months.get_loc(SPLIT_MONTH)

reproduction = {f"{EXPECTED_TEST_MONTHS} test months": len(test_months) == EXPECTED_TEST_MONTHS}
for name, expected in FIRST_RUN_SHARPE.items():
    annual_sharpe = sharpe_ratio(returns_by_name[name]) * ANNUALISING_FACTOR
    reproduction[f"{name}: annualised Sharpe ratio reproduces the first run ({expected})"] = np.isclose(
        annual_sharpe, expected, rtol=0, atol=6e-5
    )
reproduction["sub-period A: 72 months, ending 2023-12"] = (
    split_position == 72 and str(test_months[split_position - 1]) == "2023-12"
)
reproduction["sub-period B: 33 months, 2024-01 to 2026-09"] = (
    len(test_months) - split_position == 33 and str(test_months[-1]) == "2026-09"
)
run_checks(reproduction)

# ---- Part B: POST-RUN check 1, sub-periods
print("\nPOST-RUN check 1: sub-periods. Same weights as the first run; only the months measured are split.")
print("A = 2018-01 to 2023-12 (72 months); B = 2024-01 to 2026-09 (33 months). Monthly excess returns, annualised.")
print(f"Intervals: formula and paired bootstrap ({REPLICATIONS:,} resamples, seed {SEED}); the wider is quoted.")

print(f"\n{'Sharpe ratio':<22}{'all':>10}{'A':>10}{'B':>10}")
for name, series in returns_by_name.items():
    part_a, part_b = split_periods(series, split_position)
    sharpes = [sharpe_ratio(part) * ANNUALISING_FACTOR for part in (series, part_a, part_b)]
    print(f"{name:<22}{sharpes[0]:>10.4f}{sharpes[1]:>10.4f}{sharpes[2]:>10.4f}")

print("\nGap = first minus second. SDG = smallest detectable gap, from the larger SE. B-A = (gap in B) minus (gap in A).")
print(f"{'comparison':<22}{'period':<7}{'gap':>8}{'formula SE':>12}{'boot. SE':>10}{'SDG':>8}   {'wider 95% interval':<20}zero inside")
for label, first_name, second_name in comparisons:
    first_all = returns_by_name[first_name]
    second_all = returns_by_name[second_name]
    first_a, first_b = split_periods(first_all, split_position)
    second_a, second_b = split_periods(second_all, split_position)

    summary_all = gap_summary(first_all, second_all)
    summary_a = gap_summary(first_a, second_a)
    summary_b = gap_summary(first_b, second_b)
    summary_difference = difference_of_gaps(first_a, second_a, first_b, second_b, summary_a, summary_b)

    print_row(label, "all", summary_all)
    print_row(label, "A", summary_a)
    print_row(label, "B", summary_b)
    print_row(label, "B-A", summary_difference)
    print()

# ---- Part C: POST-RUN check 2, r_f = 0 (raw monthly returns, weights re-estimated on them)
RAW_FILE = "data/monthly_returns_2008-01_to_2026-09.csv"
RF_FILE = "data/monthly_rf_2008-01_to_2026-09.csv"

# Same layout as the excess-return file, so the same loader works (its name reflects its first use)
raw_returns = load_excess_returns(RAW_FILE)
risk_free = load_excess_returns(RF_FILE)["rf_monthly"]

run_checks({
    "raw file has the same months and stocks as the excess file":
        raw_returns.index.equals(excess_returns.index) and list(raw_returns.columns) == list(excess_returns.columns),
    "raw returns minus risk-free equals the excess returns, every month and stock":
        np.allclose(raw_returns.sub(risk_free, axis=0), excess_returns, rtol=0, atol=1e-12),
})

raw_weights_by_name, raw_returns_by_name = build_strategies(raw_returns, FIRST_TEST_MONTH, WINDOW_LENGTHS)

print("\nPOST-RUN check 2: r_f = 0. Raw monthly returns, weights re-estimated on raw returns, same 105 test months.")
average_risk_free = risk_free.loc[test_months].mean() * MONTHS_PER_YEAR
print(f"Average risk-free rate over the test months: {average_risk_free:.4f} a year")

print(f"\n{'Sharpe ratio, annualised':<26}{'first run':>11}{'r_f = 0':>10}{'change':>9}{'vol (r_f = 0)':>15}")
for name in returns_by_name:
    first = sharpe_ratio(returns_by_name[name]) * ANNUALISING_FACTOR
    zero = sharpe_ratio(raw_returns_by_name[name]) * ANNUALISING_FACTOR
    volatility = raw_returns_by_name[name].std() * ANNUALISING_FACTOR
    print(f"{name:<26}{first:>11.4f}{zero:>10.4f}{zero - first:>+9.4f}{volatility:>15.4f}")

print(f"\n{'comparison':<22}{'basis':<7}{'gap':>8}{'formula SE':>12}{'boot. SE':>10}{'SDG':>8}   {'wider 95% interval':<20}zero inside")
for label, first_name, second_name in comparisons:
    first_summary = gap_summary(returns_by_name[first_name], returns_by_name[second_name])
    zero_summary = gap_summary(raw_returns_by_name[first_name], raw_returns_by_name[second_name])
    print_row(label, "first", first_summary)
    print_row(label, "rf = 0", zero_summary)
    print(f"{'':<22}{'change':<7}{zero_summary['gap'] - first_summary['gap']:>+8.3f}")
    print()

print("Weights re-estimated on raw returns against the first run's weights:")
for window_length in WINDOW_LENGTHS:
    name = f"M = {window_length}"
    distance = (raw_weights_by_name[name] - weights_by_name[name]).abs().sum(axis=1).mean()
    print(f"  {name}: average monthly distance {distance:.3f}   (0 = identical weights, 2 = a complete switch)")

# ---- Part D: POST-RUN check 3, buy-and-hold 1/N (equal weights in the first test month, no trading afterwards)
test_raw = raw_returns.loc[test_months]
# value of 1 put into each stock at the start, as it stands before each month's return
growth_before_month = (1 + test_raw).cumprod().shift(1).fillna(1.0)
buy_hold_weights = growth_before_month.div(growth_before_month.sum(axis=1), axis=0)
buy_hold_raw = (buy_hold_weights * test_raw).sum(axis=1)
buy_hold_excess = buy_hold_raw - risk_free.loc[test_months]

run_checks({
    "buy-and-hold weights start at 1/N and add up to 1 every month":
        np.allclose(buy_hold_weights.iloc[0], 1 / test_raw.shape[1]) and np.allclose(buy_hold_weights.sum(axis=1), 1.0),
    "buy-and-hold final wealth equals the average of the four stocks' final wealth":
        np.isclose((1 + buy_hold_raw).prod(), (1 + test_raw).prod().mean()),
})

print("\nPOST-RUN check 3: buy-and-hold 1/N. Equal weights in 2018-01, no trading afterwards; weights drift with raw returns.")
rebalanced_sharpe = sharpe_ratio(returns_by_name["1/N"]) * ANNUALISING_FACTOR
buy_hold_sharpe = sharpe_ratio(buy_hold_excess) * ANNUALISING_FACTOR
print(f"Sharpe ratio, annualised: rebalanced 1/N {rebalanced_sharpe:.4f}, buy-and-hold {buy_hold_sharpe:.4f}, change {buy_hold_sharpe - rebalanced_sharpe:+.4f}")
final_weights = ", ".join(f"{stock} {weight:.3f}" for stock, weight in buy_hold_weights.iloc[-1].items())
print(f"Buy-and-hold weights in the last test month: {final_weights}")

print("\nB&H = buy-and-hold. Gap = first minus second.")
print(f"{'comparison':<22}{'period':<7}{'gap':>8}{'formula SE':>12}{'boot. SE':>10}{'SDG':>8}   {'wider 95% interval':<20}zero inside")
buy_hold_comparisons = [
    ("M = 60 minus B&H", returns_by_name["M = 60"], buy_hold_excess),
    ("M = 120 minus B&H", returns_by_name["M = 120"], buy_hold_excess),
    ("B&H minus 1/N", buy_hold_excess, returns_by_name["1/N"]),
]
for label, first_series, second_series in buy_hold_comparisons:
    print_row(label, "all", gap_summary(first_series, second_series))


# ---- Part E: POST-RUN check 4, turnover with drift
def monthly_trades(weights, returns):
    # Sum of absolute trades in each month: from last month's weights, after they have drifted with
    # that month's returns (raw returns, not excess), to this month's target weights
    grown = weights * (1 + returns.loc[weights.index])
    drifted = grown.div(grown.sum(axis=1), axis=0)
    return (weights - drifted.shift(1)).abs().sum(axis=1).iloc[1:]


toy_weights = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], columns=["a", "b"])
toy_returns = pd.DataFrame([[0.10, -0.10], [0.0, 0.0]], columns=["a", "b"])
toy_trades = monthly_trades(toy_weights, toy_returns)
run_checks({
    "monthly_trades on the toy case: one month, trade |0.5 - 0.55| + |0.5 - 0.45| = 0.10":
        len(toy_trades) == 1 and np.isclose(toy_trades.iloc[0], 0.10),
})

print("\nPOST-RUN check 4: average monthly turnover = sum of absolute trades (a complete switch between two stocks = 2.0)")
print(f"{'':<10}{'target to target':>18}{'with drift':>12}")
for name, weights in weights_by_name.items():
    target_to_target = monthly_turnover(weights)
    with_drift = monthly_trades(weights, raw_returns).mean()
    print(f"{name:<10}{target_to_target:>18.4f}{with_drift:>12.4f}")

# ---- Part F: one figure with the locked result and every post-run check, each row labelled
FIGURE_FOLDER = Path("figures/week8")
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LOCKED_COLOUR = "#2a78d6"       # the locked result
POST_RUN_COLOUR = INK_SECONDARY  # every post-run row, kept quiet


def check_rows(label):
    # The four rows of one comparison: the locked result, then three post-run checks
    return [
        (label, "all", "locked result, 105 months", "locked"),
        (label, "A", "post-run: 2018-01 to 2023-12", "post-run"),
        (label, "B", "post-run: 2024-01 to 2026-09", "post-run"),
        (label, "rf = 0", "post-run: r_f = 0, raw returns", "post-run"),
    ]


figure_groups = [
    ("M = 60 minus 1/N", check_rows("M = 60 gap")),
    ("M = 120 minus 1/N", check_rows("M = 120 gap")),
    ("M = 120 minus M = 60", check_rows("M = 120 minus M = 60")),
    ("Buy-and-hold 1/N (B&H), post-run", [
        ("M = 60 minus B&H", "all", "post-run: M = 60 minus B&H", "post-run"),
        ("M = 120 minus B&H", "all", "post-run: M = 120 minus B&H", "post-run"),
        ("B&H minus 1/N", "all", "post-run: B&H minus 1/N", "post-run"),
    ]),
]


def plot_all_gaps(groups, filename):
    summaries = {(label, period): summary for label, period, summary in printed_rows}

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    row_positions = []
    row_texts = []
    seen_kinds = set()
    y = 0.0
    for title, rows in groups:
        # group heading: x in axes units (0 = left edge), y in data units
        ax.text(0.01, y, title, transform=ax.get_yaxis_transform(), va="center",
                fontsize=9, fontweight="bold", color=INK)
        for label, period, text, kind in rows:
            y += 1.0
            summary = summaries[(label, period)]
            gap = summary["gap"]
            low, high = summary["quoted_interval"]
            locked = kind == "locked"
            legend_label = None
            if kind not in seen_kinds:
                seen_kinds.add(kind)
                legend_label = "Locked result (12 October 2026)" if locked else "Post-run check (13 October 2026)"
            ax.errorbar(
                gap, y,
                xerr=[[max(gap - low, 0.0)], [max(high - gap, 0.0)]],
                fmt="o" if locked else "s",
                color=LOCKED_COLOUR if locked else POST_RUN_COLOUR,
                elinewidth=1.6, capsize=3, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=legend_label,
            )
            row_positions.append(y)
            row_texts.append(text)
        y += 1.8  # air before the next group

    ax.axvline(0, color=INK_SECONDARY, linewidth=0.9)
    ax.set_yticks(row_positions)
    ax.set_yticklabels(row_texts, fontsize=8, color=INK_SECONDARY)
    ax.set_ylim(row_positions[-1] + 0.8, -0.8)  # larger value first: the first group sits at the top
    ax.set_xlabel("Sharpe-ratio gap, annualised (first minus second); bars are 95% intervals",
                  fontsize=9, color=INK_SECONDARY)
    ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)

    title_text = (
        "Sharpe-ratio gaps with 95% intervals: the locked result and the post-run checks\n"
        "monthly excess returns (r_f = 0 rows: raw returns), 2018-01 to 2026-09 (105 months), long-only, rebalanced monthly, no costs\n"
        f"bars: the wider of the formula and paired-bootstrap intervals ({REPLICATIONS:,} resamples, seed {SEED})"
    )
    fig.suptitle(title_text, x=0.01, y=0.99, ha="left", va="top", fontsize=9, color=INK)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(FIGURE_FOLDER / filename, dpi=200)
    plt.close(fig)
    print(f"saved {FIGURE_FOLDER / filename}")


print()
plot_all_gaps(figure_groups, "post_run_gaps.png")