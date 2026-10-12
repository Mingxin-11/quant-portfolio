import numpy as np
import pandas as pd

from portfolio_tools import (
    chance_realised_gap_positive,
    months_needed_to_detect,
    power_to_detect,
    sharpe_gap_se,
    smallest_detectable_gap,
)

TEST_MONTHS = 105                               # January 2018 to September 2026
SHORT_TEST_MONTHS = 72                          # the original design: January 2018 to December 2023
MONTHS_PER_YEAR = 12
ANNUALISING_FACTOR = np.sqrt(MONTHS_PER_YEAR)   # monthly Sharpe ratio, or its SE, to a yearly figure
SIGNIFICANCE_LEVEL = 0.05                       # two-sided; fixed now, before any run
TARGET_POWER = 0.80                             # fixed now, before any run
GAP_WITH_EDGE = 0.10                            # rounded size of the Step 4 gaps in truth 1 (+0.09 to +0.12)
GAP_WITHOUT_EDGE = -0.03                        # rounded size of the Step 4 gaps in truth 2 (-0.04 to -0.02)

# the toy case for the hand check (annualised Sharpe units): standard error 0.20 after 100 months, true gap 0.10
HAND_STANDARD_ERROR = 0.20
HAND_MONTHS = 100
HAND_GAP = 0.10
HAND_RELATIVE_TOLERANCE = 0.005                 # "agrees" means within 0.5%
SMALLEST_GAP_BY_HAND = 0.560                      # YOUR VALUE: smallest detectable gap at the toy standard error (three significant figures)
POWER_BY_HAND = 0.079                             # YOUR VALUE: power at the toy true gap (three significant figures)
CHANCE_POSITIVE_BY_HAND = 0.692                  # YOUR VALUE: chance that the realised gap is above 0 at the toy true gap (three significant figures)
MONTHS_BY_HAND = 3140                            # YOUR VALUE: months after which the toy true gap is the smallest detectable gap (three significant figures)

excess_returns = pd.read_csv(
    "data/monthly_excess_returns_2008-01_to_2026-09.csv",
    index_col=0,
    parse_dates=True,
).to_period("M")

planning_returns = excess_returns.loc["2008-01":"2017-12"]
equal_weight_returns = planning_returns.mean(axis=1)
equal_sharpe = equal_weight_returns.mean() / equal_weight_returns.std()

# one row per stock against 1/N: (ticker, correlation with 1/N, SE of the Sharpe gap at 72 months, SE at 105 months), SEs annualised
anchors = []
for ticker in planning_returns.columns:
    stock_returns = planning_returns[ticker]
    stock_sharpe = stock_returns.mean() / stock_returns.std()
    correlation = stock_returns.corr(equal_weight_returns)
    se_72 = sharpe_gap_se(stock_sharpe, equal_sharpe, correlation, SHORT_TEST_MONTHS) * ANNUALISING_FACTOR
    se_105 = sharpe_gap_se(stock_sharpe, equal_sharpe, correlation, TEST_MONTHS) * ANNUALISING_FACTOR
    anchors.append((ticker, correlation, se_72, se_105))

print("Part A: your four functions against four values worked by hand (annualised Sharpe units; 5% level, 80% power)")
hand_cases = [
    # (what is computed, function value, YOUR HAND VALUE)
    (f"smallest detectable gap, SE {HAND_STANDARD_ERROR:.2f}", smallest_detectable_gap(0.2, 0.05, 0.8), SMALLEST_GAP_BY_HAND),
    (f"power, true gap {HAND_GAP:.2f}, SE {HAND_STANDARD_ERROR:.2f}", power_to_detect(HAND_GAP, HAND_STANDARD_ERROR, SIGNIFICANCE_LEVEL), POWER_BY_HAND),
    (f"chance realised gap > 0, true gap {HAND_GAP:.2f}, SE {HAND_STANDARD_ERROR:.2f}", chance_realised_gap_positive(HAND_GAP, HAND_STANDARD_ERROR), CHANCE_POSITIVE_BY_HAND),
    (f"months until a gap of {HAND_GAP:.2f} is detectable", months_needed_to_detect(HAND_GAP, HAND_MONTHS, HAND_STANDARD_ERROR, SIGNIFICANCE_LEVEL, TARGET_POWER), MONTHS_BY_HAND),
]
hand_agrees = []
for description, function_value, hand_value in hand_cases:
    hand_agrees.append(np.isclose(function_value, hand_value, rtol=HAND_RELATIVE_TOLERANCE, atol=0))
    print(f"   {description:52} | function {function_value:10.4f} | by hand {hand_value:10.4f} | agree: {hand_agrees[-1]}")
print("   the functions agree with every hand value:", all(hand_agrees))

print("Part B: the smallest gap a test could detect (80% power, 5% level), annualised Sharpe units; planning months only")
print("   anchor | corr with 1/N | SE, 72 months | smallest gap, 72 | SE, 105 months | smallest gap, 105")
for ticker, correlation, se_72, se_105 in anchors:
    smallest_72 = smallest_detectable_gap(se_72, SIGNIFICANCE_LEVEL, TARGET_POWER)
    smallest_105 = smallest_detectable_gap(se_105, SIGNIFICANCE_LEVEL, TARGET_POWER)
    print(f"   {ticker:6} | {correlation:13.3f} | {se_72:13.3f} | {smallest_72:16.3f} | {se_105:14.3f} | {smallest_105:17.3f}")

print(f"Part C: a 105-month test if the true gap were {GAP_WITH_EDGE:+.2f} (like truth 1) or {GAP_WITHOUT_EDGE:+.2f} (like truth 2)")
print(f"   anchor | power at {GAP_WITH_EDGE:+.2f} | chance the realised gap is above 0 | power at {GAP_WITHOUT_EDGE:+.2f} | chance the realised gap is above 0")
for ticker, correlation, se_72, se_105 in anchors:
    power_with_edge = power_to_detect(GAP_WITH_EDGE, se_105, SIGNIFICANCE_LEVEL)
    power_without_edge = power_to_detect(GAP_WITHOUT_EDGE, se_105, SIGNIFICANCE_LEVEL)
    positive_with_edge = chance_realised_gap_positive(GAP_WITH_EDGE, se_105)
    positive_without_edge = chance_realised_gap_positive(GAP_WITHOUT_EDGE, se_105)
    print(f"   {ticker:6} | {power_with_edge:14.3f} | {positive_with_edge:34.3f} | {power_without_edge:14.3f} | {positive_without_edge:34.3f}")

print("Part D: years of testing after which a true gap would be the smallest detectable gap")
print(f"   anchor | gap {GAP_WITH_EDGE:.2f}, counted from 72 months | gap {GAP_WITH_EDGE:.2f}, counted from 105 months | gap {abs(GAP_WITHOUT_EDGE):.2f}, counted from 105 months")
years_agree = []
for ticker, correlation, se_72, se_105 in anchors:
    years_from_72 = months_needed_to_detect(GAP_WITH_EDGE, SHORT_TEST_MONTHS, se_72, SIGNIFICANCE_LEVEL, TARGET_POWER) / MONTHS_PER_YEAR
    years_from_105 = months_needed_to_detect(GAP_WITH_EDGE, TEST_MONTHS, se_105, SIGNIFICANCE_LEVEL, TARGET_POWER) / MONTHS_PER_YEAR
    years_small_gap = months_needed_to_detect(GAP_WITHOUT_EDGE, TEST_MONTHS, se_105, SIGNIFICANCE_LEVEL, TARGET_POWER) / MONTHS_PER_YEAR
    years_agree.append(np.isclose(years_from_72, years_from_105))
    print(f"   {ticker:6} | {years_from_72:32.1f} | {years_from_105:33.1f} | {years_small_gap:33.1f}")

print("Part E: logic checks on the four functions")
smallest_gap = smallest_detectable_gap(HAND_STANDARD_ERROR, SIGNIFICANCE_LEVEL, TARGET_POWER)
months_at_smallest_gap = months_needed_to_detect(smallest_gap, TEST_MONTHS, HAND_STANDARD_ERROR, SIGNIFICANCE_LEVEL, TARGET_POWER)
months_at_half_gap = months_needed_to_detect(smallest_gap / 2, TEST_MONTHS, HAND_STANDARD_ERROR, SIGNIFICANCE_LEVEL, TARGET_POWER)
print("   power with no true gap equals the significance level:", np.isclose(power_to_detect(0.0, HAND_STANDARD_ERROR, SIGNIFICANCE_LEVEL), SIGNIFICANCE_LEVEL))
print("   with no true gap the realised gap is above zero half the time:", np.isclose(chance_realised_gap_positive(0.0, HAND_STANDARD_ERROR), 0.5))
print("   a positive true gap makes a positive realised gap more likely than not:", chance_realised_gap_positive(HAND_GAP, HAND_STANDARD_ERROR) > 0.5)
print("   power at the smallest detectable gap equals the target power:", np.isclose(power_to_detect(smallest_gap, HAND_STANDARD_ERROR, SIGNIFICANCE_LEVEL), TARGET_POWER, atol=1e-5))
print("   months needed for the smallest detectable gap equal the months already tested:", np.isclose(months_at_smallest_gap, TEST_MONTHS))
print("   half that gap needs four times as many months:", np.isclose(months_at_half_gap, 4 * TEST_MONTHS))
print("   years needed, counted from 72 or from 105 months, agree for every anchor:", all(years_agree))