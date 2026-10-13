import numpy as np
import pandas as pd
from portfolio_tools import sharpe_gap_se

TEST_MONTHS = 105                          # January 2018 to September 2026
FIRST_TEST_MONTH = pd.Period("2018-01")
SHORT_TEST_MONTHS = 72                     # the original design: January 2018 to December 2023
ANNUALISING_FACTOR = np.sqrt(12)           # monthly Sharpe ratio (or its SE) times this = yearly
HAND_TOLERANCE = 0.00005                   # Part D: "agree" means closer than this
TABLE_TOLERANCE = 0.001                    # Part E: the same, for one table value checked by hand
JPM_FIRST_TERM_BY_HAND = 0.162

excess_returns = pd.read_csv("data/monthly_excess_returns_2008-01_to_2026-09.csv", index_col=0, parse_dates=True).to_period("M")

planning_returns = excess_returns.loc["2008-01":"2017-12"]
print("Part A: the planning table")
print("   months:", len(planning_returns), "| first:", planning_returns.index[0], "| last:", planning_returns.index[-1])
print("   every planning month is before the first test month:", planning_returns.index.max() < FIRST_TEST_MONTH)

equal_weight_returns = planning_returns.mean(axis=1)
equal_sd = equal_weight_returns.std()

print("Part B: sd columns: 120 planning months; SE columns: for a 105-month test; all in % a month")
print("   ticker | corr with 1/N | sd of difference: direct | from parts | SE of mean gap: paired | pairing ignored")
routes_agree = []
for ticker in planning_returns.columns:
    stock_returns = planning_returns[ticker]
    stock_sd = stock_returns.std()
    difference = stock_returns - equal_weight_returns
    direct_sd = difference.std()
    correlation = stock_returns.corr(equal_weight_returns)
    parts_sd = np.sqrt(
        stock_sd**2
        + equal_sd**2
        - 2 * correlation * stock_sd * equal_sd
    )
    # Standard error of the mean gap (paired vs unpaired):
    paired_se = direct_sd / np.sqrt(TEST_MONTHS)
    ignored_se = np.sqrt(stock_sd**2 / TEST_MONTHS + equal_sd**2 / TEST_MONTHS)
    routes_agree.append(np.isclose(direct_sd, parts_sd))
    print(f"   {ticker:6} | {correlation:13.3f} | {100 * direct_sd:23.3f} | {100 * parts_sd:10.3f} | {100 * paired_se:22.3f} | {100 * ignored_se:15.3f}")
print("   the two routes agree for every stock:", all(routes_agree))

print("Part C: the third route, d'Sigma d")
cov_matrix = planning_returns.cov().to_numpy()
third_route_agrees = []
for position, ticker in enumerate(planning_returns.columns):
    n_assets = len(planning_returns.columns)
    weight_difference = np.zeros(n_assets)
    weight_difference[position] = 1.0
    weight_difference -= 1.0 / n_assets

    # Quadratic form: d' * Sigma * d (using sample variance ddof=1)
    variance_from_weights = float(
        weight_difference @ cov_matrix @ weight_difference
    )
    difference = planning_returns[ticker] - equal_weight_returns
    third_route_agrees.append(np.isclose(variance_from_weights, difference.var()))
    print(f"   {ticker:6} | weights sum to {weight_difference.sum():.1f} | d'Sigma d = {variance_from_weights:.6f} | variance of the series = {difference.var():.6f}")
print("   the third route agrees for every stock:", all(third_route_agrees))


print("Part D: sharpe_gap_se against four values worked by hand (monthly units, 100 months)")
print("   which definition runs:", sharpe_gap_se.__code__.co_filename, "line", sharpe_gap_se.__code__.co_firstlineno)
print("   what one call returns:", type(sharpe_gap_se(0.2, 0.2, 0.5, 100)))
hand_cases = [
    (0.5, 0.5, 1.0, 100, 0.00000),
    (0.0, 0.0, 0.5, 100, 0.10000),
    (0.2, 0.2, 0.5, 100, 0.10149),
    (0.3, 0.1, 0.5, 100, 0.10210),
]
function_agrees = []
for sharpe_p, sharpe_b, correlation, months, hand_value in hand_cases:
    function_value = sharpe_gap_se(sharpe_p, sharpe_b, correlation, months)
    function_agrees.append(abs(function_value - hand_value) < HAND_TOLERANCE)
    print(f"   SR {sharpe_p} and {sharpe_b} | corr {correlation} | function {function_value:.5f} | by hand {hand_value:.5f} | agree: {function_agrees[-1]}")
print("   the function agrees with every hand value:", all(function_agrees))

print("Part E: how precisely could each stock's Sharpe ratio be separated from 1/N's? Planning months only; SEs annualised")
equal_sharpe = equal_weight_returns.mean() / equal_weight_returns.std()
expected_ratio = np.sqrt(SHORT_TEST_MONTHS / TEST_MONTHS)
print("   1/N planning Sharpe ratio (monthly):", round(equal_sharpe, 3))
print("   ticker | Sharpe (monthly) | corr with 1/N | SE, 105 months | SE, 72 months | first term only, 105 | full / first term | ratio 105/72")
ratio_is_right = []
for ticker in planning_returns.columns:
    stock_returns = planning_returns[ticker]
    stock_sharpe = stock_returns.mean() / stock_returns.std()
    correlation = stock_returns.corr(equal_weight_returns)
    se_105 = np.sqrt(12) * sharpe_gap_se(stock_sharpe, equal_sharpe, correlation, TEST_MONTHS)
    se_72 = np.sqrt(12) * sharpe_gap_se(stock_sharpe, equal_sharpe, correlation, SHORT_TEST_MONTHS)
    # First term only (simplified approximation): sqrt(12 * 2 * (1 - correlation) / months)
    first_term_105 = np.sqrt(12 * 2 * (1 - correlation) / TEST_MONTHS)
    full_over_first = se_105 / first_term_105
    ratio = se_105 / se_72
    ratio_is_right.append(np.isclose(ratio, expected_ratio))
    if ticker == "JPM":
        jpm_first_term = first_term_105
    print(f"   {ticker:6} | {stock_sharpe:16.3f} | {correlation:13.3f} | {se_105:14.3f} | {se_72:13.3f} | {first_term_105:20.3f} | {full_over_first:17.3f} | {ratio:12.4f}")
print("   every ratio equals sqrt(72/105):", all(ratio_is_right))
print(f"   so going from 72 to 105 months makes the standard error {100 * (1 - expected_ratio):.1f}% narrower, for every row")
jpm_row_agrees = abs(jpm_first_term - JPM_FIRST_TERM_BY_HAND) < TABLE_TOLERANCE
print(f"   hand check, JPM first term only: table {jpm_first_term:.3f} | by hand {JPM_FIRST_TERM_BY_HAND:.3f} | agree: {jpm_row_agrees}")