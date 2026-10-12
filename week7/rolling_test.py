import pandas as pd
import numpy as np
import scipy
from portfolio_tools import (
    exact_long_only_weights, 
    slsqp_weights, 
    exact_rule, 
    portfolio_returns, 
    rolling_weights
)


EXCESS_FILE = "data/monthly_excess_returns_2008-01_to_2026-09.csv"
FIRST_TEST_MONTH = "2018-01"
WINDOW_LENGTHS = [60, 120]

excess_returns = pd.read_csv(EXCESS_FILE, index_col=0, parse_dates=True).to_period("M")

first_test_position = excess_returns.index.get_loc(FIRST_TEST_MONTH)
number_of_test_months = len(excess_returns) - first_test_position

print("Table:", excess_returns.shape)
print("First test month", FIRST_TEST_MONTH, "sits at position", first_test_position)
print("Test months:", number_of_test_months)

for window_length in WINDOW_LENGTHS:
    # Extract the historical training window of length M ending the month before the test month
    window = excess_returns.iloc[first_test_position - window_length : first_test_position]
    print("M =", window_length, "| rows:", len(window), "| from", window.index[0], "to", window.index[-1])

def equal_weight_rule(window):
    """1/N: the same weight for every stock, whatever the window says."""
    number_of_stocks = window.shape[1]
    return np.full(number_of_stocks, 1 / number_of_stocks)


print("Part A: 1/N through the rolling loop")
for M in WINDOW_LENGTHS:
    weights = rolling_weights(excess_returns, M, first_test_position, equal_weight_rule)
    print("M =", M, "| weights table:", weights.shape)
    print("   first and last test month:", weights.index[0], weights.index[-1])
    print("   every row sums to 1:", np.allclose(weights.sum(axis=1), 1))

def last_row_rule(window):
    """Test probe: hand back the window's final row, to see which month it is."""
    return window.iloc[-1].to_numpy()


def first_row_rule(window):
    """Test probe: hand back the window's first row."""
    return window.iloc[0].to_numpy()


print("Part B: probes")
for M in WINDOW_LENGTHS:
    last_seen = rolling_weights(excess_returns, M, first_test_position, last_row_rule)
    first_seen = rolling_weights(excess_returns, M, first_test_position, first_row_rule)
    previous_month = excess_returns.shift(1).iloc[first_test_position:]
    month_M_back = excess_returns.shift(M).iloc[first_test_position:]
    print("M =", M)
    print("   window ends the month before the test month:", np.allclose(last_seen, previous_month))
    print("   window starts M months before the test month:", np.allclose(first_seen, month_M_back))


def slsqp_rule(window):
    """Same job as exact_rule, done by the independent numerical optimiser."""
    mean_returns = window.mean().to_numpy()
    cov_matrix = window.cov().to_numpy()
    return slsqp_weights(mean_returns, cov_matrix)

print("Part C: the exact optimiser through the rolling loop")
for M in WINDOW_LENGTHS:
    exact_weights = rolling_weights(excess_returns, M, first_test_position, exact_rule)
    print("M =", M, "| weights table:", exact_weights.shape)
    print("   no negative weight:", (exact_weights >= 0).all().all())
    print("   every row sums to 1:", np.allclose(exact_weights.sum(axis=1), 1))
    print("   no missing values:", exact_weights.notna().all().all())

    first_window = excess_returns.iloc[first_test_position - M : first_test_position]
    direct_call = exact_long_only_weights(first_window.mean().to_numpy(), first_window.cov().to_numpy())
    print("   first row equals a direct call on the first window:", np.allclose(exact_weights.iloc[0], direct_call))

print("Part D: exact solver against SLSQP, every window")
print("SciPy version:", scipy.__version__)
for M in WINDOW_LENGTHS:
    exact_weights = rolling_weights(excess_returns, M, first_test_position, exact_rule)
    slsqp_table = rolling_weights(excess_returns, M, first_test_position, slsqp_rule)
    gap_per_window = (exact_weights - slsqp_table).abs().max(axis=1)

    print("M =", M, "| windows compared:", len(gap_per_window))
    print("   largest gap over all windows:", gap_per_window.max())
    print("   test month with the largest gap:", gap_per_window.idxmax())
    print("   windows with a gap above 1e-4:", (gap_per_window > 1e-4).sum())
    print("   exact weights change from month to month:", (exact_weights.nunique() > 1).any())
    print("   SLSQP weights change from month to month:", (slsqp_table.nunique() > 1).any())

SHOCKED_MONTH = "2022-03"
NEXT_MONTH = "2022-04"

print("Part E: look-ahead test with the real rule (exact optimiser)")
for M in WINDOW_LENGTHS:
    original_weights = rolling_weights(
        excess_returns, M, first_test_position, exact_rule
    )
    biggest_holding = original_weights.loc[NEXT_MONTH].idxmax()

    # Create independent copy and apply -0.90 shock
    shocked_returns = excess_returns.copy()
    shocked_returns.loc[SHOCKED_MONTH, biggest_holding] -= 0.90

    shocked_weights = rolling_weights(
        shocked_returns, M, first_test_position, exact_rule
    )
    row_changed = (original_weights != shocked_weights).any(axis=1)

    print("M =", M)
    print(
        "   original table untouched by the shock:",
        excess_returns.loc[SHOCKED_MONTH, biggest_holding]
        != shocked_returns.loc[SHOCKED_MONTH, biggest_holding],
    )
    print(
        "   weights up to and including the shocked month unchanged:",
        not row_changed.loc[:SHOCKED_MONTH].any(),
    )
    print(
        "   weights for the next month changed:",
        row_changed.loc[NEXT_MONTH],
    )


toy_returns = pd.DataFrame(
    {"A": [0.10, -0.10, 0.00], "B": [0.20, 0.05, 0.30]},
    index=pd.period_range("2000-01", periods=3, freq="M"),
)
toy_weights = pd.DataFrame(
    {"A": [0.50, 0.25], "B": [0.50, 0.75]},
    index=pd.period_range("2000-02", periods=2, freq="M"),
)
expected = [-0.025, 0.225]

def same_numbers(result, expected):
    """True only if both have the same length and the same values (to rounding)."""
    return len(result) == len(expected) and np.allclose(result, expected)


print("Part F: weights times returns, toy table with a known answer")
result = portfolio_returns(toy_weights, toy_returns)
print("   months returned:", list(result.index))
print("   values match the hand calculation:", same_numbers(result, expected))
late = portfolio_returns(toy_weights, toy_returns.shift(1))
print("   a one-month-late multiplication does NOT match:", not same_numbers(late, expected))

print("Part F: the same function on real 1/N weights (True/False only)")
equal_weights = rolling_weights(excess_returns, 60, first_test_position, equal_weight_rule)
equal_returns = portfolio_returns(equal_weights, excess_returns)
plain_average = excess_returns.loc[equal_weights.index].mean(axis=1)
print("   one return per test month:", len(equal_returns) == len(equal_weights))
print("   equals the plain average of the four stocks each month:", same_numbers(equal_returns, plain_average))