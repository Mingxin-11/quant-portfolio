# Lesson 8.3: does serial dependence in the monthly returns change how uncertain the Sharpe gaps are?
# Part A: an invented world with a known answer (nothing in Part A touches the real data).
#   Two invented series of monthly returns, each with true Sharpe ratio 0, so the true gap is exactly 0.
#   Question: how often does each method's 95% interval exclude 0 (a false alarm)? A perfect 95% interval: 5%.
# Part B: the real test months. Everything printed in Part B is post-run and is reported whatever it shows;
#   none of it replaces the locked result.
# Predictions locked 13 October 2026, 12:56 SGT; block lengths locked 13:02 SGT (project doc claude/19_WEEK8_LOCK_RECORD_8_3.md).
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "week7"))

from portfolio_tools import (
    ANNUALISING_FACTOR,
    EXCESS_FILE,
    EXPECTED_TEST_MONTHS,
    FIRST_TEST_MONTH,
    REPLICATIONS,
    SEED,
    WINDOW_LENGTHS,
    Z_95,
    annual_formula_se,
    build_strategies,
    gap_summary,
    load_excess_returns,
    paired_bootstrap_sharpe_gaps,
    run_checks,
)

# Settings of the invented world
WORLD_MONTHS = 105                     # as long as the real test period
SHOCK_CORRELATION = 0.8                # close to the real correlation between a strategy and 1/N
WARM_UP_MONTHS = 100                   # simulated and thrown away, so the series forgets its starting value
PERSISTENCE_VALUES = [0.0, 0.2, 0.4]   # phi: the share of last month's value that carries over
HISTORIES = 2000                       # invented histories per world
BOOTSTRAP_RESAMPLES = 1000             # resamples per history (the real run uses REPLICATIONS = 5000)
BLOCK_LENGTHS_TO_TEST = [1, 3, 6, 12]

# Settings of the real-data part (Part B)
PRIMARY_BLOCK_LENGTH = 6               # locked 13:02 SGT: the length that prediction 3 refers to
OTHER_BLOCK_LENGTH = 3                 # locked with it: the check on how much the answer depends on the block length
BLOCK_LENGTHS_REAL = [PRIMARY_BLOCK_LENGTH, OTHER_BLOCK_LENGTH]
MAX_LAG = 12                           # autocorrelations are computed for lags 1 to 12 months
RULE_GAP = "M = 120 minus M = 60"      # the gap that locked rule 5 is about
RULE_THRESHOLD = 0.110                 # the number in locked rule 5
GAP_PAIRS = [                          # (label, strategy a, strategy b): the gap is the Sharpe ratio of a minus that of b
    ("M = 60 minus 1/N", "M = 60", "1/N"),
    ("M = 120 minus 1/N", "M = 120", "1/N"),
    ("M = 120 minus M = 60", "M = 120", "M = 60"),
]
LOCKED_FIRST_RUN = {                   # printed in the first run of 12 October 2026 (annualised):
    "M = 60 minus 1/N": (0.221, 0.217, -0.428, 0.439),       # formula SE, bootstrap SE, quoted interval (low, high);
    "M = 120 minus 1/N": (0.226, 0.225, -0.348, 0.539),      # the quoted interval is the wider of the formula
    "M = 120 minus M = 60": (0.098, 0.095, -0.091, 0.292),   # and the bootstrap interval
}


def simulate_pair(months, persistence, shock_correlation, rng):
    """Two invented series of monthly returns with true Sharpe ratio 0 each (so the true gap is 0).
    Each month's value is `persistence` times last month's value plus a fresh shock;
    the two shocks of a month are correlated."""
    covariance = [[1.0, shock_correlation], [shock_correlation, 1.0]]
    shocks = rng.multivariate_normal([0.0, 0.0], covariance, size=WARM_UP_MONTHS + months)
    values = np.zeros_like(shocks)
    for month in range(1, len(shocks)):
        values[month] = persistence * values[month - 1] + shocks[month]
    return values[WARM_UP_MONTHS:, 0], values[WARM_UP_MONTHS:, 1]


def block_positions(months, block_length, replications, rng):
    """Month numbers for circular block resamples: each block starts at a random month and takes
    block_length consecutive months, wrapping from the last month round to the first."""
    blocks_needed = int(np.ceil(months / block_length))
    starts = rng.integers(0, months, size=(replications, blocks_needed))
    steps_inside_block = np.arange(block_length)
    positions = (starts[:, :, np.newaxis] + steps_inside_block) % months
    return positions.reshape(replications, blocks_needed * block_length)[:, :months]


def block_bootstrap_sharpe_gaps(returns_a, returns_b, block_length, replications, rng):
    """Monthly Sharpe ratio of a minus that of b, recomputed on circular block resamples.
    Block length 1 is the ordinary bootstrap."""
    a = np.asarray(returns_a)
    b = np.asarray(returns_b)
    positions = block_positions(len(a), block_length, replications, rng)
    resample_a = a[positions]
    resample_b = b[positions]
    sharpe_a = resample_a.mean(axis=1) / resample_a.std(axis=1, ddof=1)
    sharpe_b = resample_b.mean(axis=1) / resample_b.std(axis=1, ddof=1)
    return sharpe_a - sharpe_b


def judge_history(series_a, series_b, rng):
    """One invented history: its annualised gap, and for each method its annualised SE
    and whether its 95% interval excludes the true gap, 0."""
    months = len(series_a)
    sharpe_a = series_a.mean() / series_a.std(ddof=1)
    sharpe_b = series_b.mean() / series_b.std(ddof=1)
    gap = (sharpe_a - sharpe_b) * ANNUALISING_FACTOR
    correlation = np.corrcoef(series_a, series_b)[0, 1]

    formula_se = annual_formula_se(sharpe_a, sharpe_b, correlation, months)
    verdicts = {"formula": (formula_se, bool(abs(gap) > Z_95 * formula_se))}

    for block_length in BLOCK_LENGTHS_TO_TEST:
        monthly_gaps = block_bootstrap_sharpe_gaps(series_a, series_b, block_length, BOOTSTRAP_RESAMPLES, rng)
        annual_gaps = monthly_gaps * ANNUALISING_FACTOR
        low, high = np.percentile(annual_gaps, [2.5, 97.5])
        verdicts[f"block length {block_length}"] = (annual_gaps.std(ddof=1), bool(low > 0 or high < 0))
    return gap, verdicts


# --- Self-checks: the invented world and the block machinery must behave as described before anything is judged
check_rng = np.random.default_rng(SEED)
long_a, long_b = simulate_pair(100000, 0.2, SHOCK_CORRELATION, check_rng)
lag_one_correlation = np.corrcoef(long_a[:-1], long_a[1:])[0, 1]
series_correlation = np.corrcoef(long_a, long_b)[0, 1]

short_a, short_b = simulate_pair(WORLD_MONTHS, 0.2, SHOCK_CORRELATION, check_rng)
sample_positions = block_positions(WORLD_MONTHS, 6, 200, check_rng)
rotation_gaps = block_bootstrap_sharpe_gaps(short_a, short_b, WORLD_MONTHS, 200, check_rng)
ordinary_se = paired_bootstrap_sharpe_gaps(short_a, short_b, REPLICATIONS, check_rng).std(ddof=1)
block_one_se = block_bootstrap_sharpe_gaps(short_a, short_b, 1, REPLICATIONS, check_rng).std(ddof=1)

run_checks({
    "invented world: lag-1 autocorrelation of a long series is about phi (0.2)": abs(lag_one_correlation - 0.2) < 0.02,
    "invented world: the two series are correlated about as set (0.8)": abs(series_correlation - SHOCK_CORRELATION) < 0.02,
    "block positions: 200 rows of 105 months, every month number between 0 and 104": (
        sample_positions.shape == (200, WORLD_MONTHS)
        and sample_positions.min() >= 0
        and sample_positions.max() <= WORLD_MONTHS - 1
    ),
    "inside a block each month follows the previous one, wrapping round the end": bool(
        np.all((sample_positions[:, 1:6] - sample_positions[:, 0:5]) % WORLD_MONTHS == 1)
    ),
    "a block as long as the sample only rotates the data: every resample gives the same gap": bool(
        np.allclose(rotation_gaps, rotation_gaps[0])
    ),
    "block length 1 gives the same SE as the ordinary bootstrap (within 10%)": abs(block_one_se / ordinary_se - 1) < 0.10,
})

# --- The experiment
rng = np.random.default_rng(SEED)
method_names = ["formula"] + [f"block length {length}" for length in BLOCK_LENGTHS_TO_TEST]

for persistence in PERSISTENCE_VALUES:
    gaps = []
    reported_se = {name: [] for name in method_names}
    false_alarms = {name: 0 for name in method_names}

    for history in range(HISTORIES):
        series_a, series_b = simulate_pair(WORLD_MONTHS, persistence, SHOCK_CORRELATION, rng)
        gap, verdicts = judge_history(series_a, series_b, rng)
        gaps.append(gap)
        for name, (se, excludes_zero) in verdicts.items():
            reported_se[name].append(se)
            false_alarms[name] += excludes_zero

    actual_sd = np.std(gaps, ddof=1)
    print(f"\nWorld with phi = {persistence}: true gap 0, {HISTORIES} invented histories of {WORLD_MONTHS} months.")
    print(f"Actual spread of the gap across histories (the SE a perfect method would report): {actual_sd:.3f}")
    print(f"{'method':<18}{'average SE reported':>21}{'reported / actual':>19}{'false alarms':>14}")
    for name in method_names:
        average_se = np.mean(reported_se[name])
        print(f"{name:<18}{average_se:>21.3f}{average_se / actual_sd:>19.3f}{false_alarms[name] / HISTORIES:>14.1%}")

print("\nBlock length 1 is the ordinary bootstrap. A false alarm is a 95% interval that excludes the true gap, 0.")
print("A perfect 95% interval gives 5%. With 2,000 histories, read each rate as plus or minus about 1 percentage point.")


# =====================================================================================================
# Part B: the real test months. Everything printed from here on is post-run.
# =====================================================================================================
def autocorrelations(series, max_lag):
    """Autocorrelations at lags 1 to max_lag: how closely a month's value moves with the value `lag` months earlier."""
    values = np.asarray(series)
    centred = values - values.mean()
    total = np.sum(centred ** 2)
    return np.array([np.sum(centred[lag:] * centred[:-lag]) / total for lag in range(1, max_lag + 1)])


def dependence_factor(autocorrs, block_length):
    """Factor by which dependence scales the variance of an average, as a block bootstrap of this block length sees it:
    1 + 2 * (sum over lags 1 to block_length - 1 of (1 - lag / block_length) * autocorrelation at that lag)."""
    return 1 + 2 * sum((1 - lag / block_length) * autocorrs[lag - 1] for lag in range(1, block_length))


def block_summary(returns_a, returns_b, block_length, replications, rng):
    """Annualised SE and 95% percentile interval of the Sharpe gap (a minus b) over circular block resamples."""
    monthly_gaps = block_bootstrap_sharpe_gaps(returns_a, returns_b, block_length, replications, rng)
    annual_gaps = monthly_gaps * ANNUALISING_FACTOR
    low, high = np.percentile(annual_gaps, [2.5, 97.5])
    return annual_gaps.std(ddof=1), (low, high)


def annual_interval_text(interval):
    low, high = interval
    return f"[{low:+.3f}, {high:+.3f}]"


excess_returns = load_excess_returns(EXCESS_FILE)
_, returns_by_name = build_strategies(excess_returns, FIRST_TEST_MONTH, WINDOW_LENGTHS)   # the weights are not needed here
test_months = len(returns_by_name["1/N"])
first_month = returns_by_name["1/N"].index[0]
last_month = returns_by_name["1/N"].index[-1]

# The six series whose dependence is looked at: three strategies, and the three monthly differences behind the three gaps
monthly_series = {
    "60": returns_by_name["M = 60"],
    "120": returns_by_name["M = 120"],
    "1/N": returns_by_name["1/N"],
    "60-1/N": returns_by_name["M = 60"] - returns_by_name["1/N"],
    "120-1/N": returns_by_name["M = 120"] - returns_by_name["1/N"],
    "120-60": returns_by_name["M = 120"] - returns_by_name["M = 60"],
}
acf_by_series = {label: autocorrelations(series, MAX_LAG) for label, series in monthly_series.items()}
band = Z_95 / np.sqrt(test_months)

# The locked first-run rows recomputed with the shared tools, and the block-length-1 SE of each gap
locked_by_gap = {}
block_one_se_by_gap = {}
for label, name_a, name_b in GAP_PAIRS:
    locked_by_gap[label] = gap_summary(returns_by_name[name_a], returns_by_name[name_b])
    block_one_se_by_gap[label] = block_summary(
        returns_by_name[name_a], returns_by_name[name_b], 1, REPLICATIONS, np.random.default_rng(SEED)
    )[0]

locked_run_reproduced = True
block_one_matches = True
for label in locked_by_gap:
    summary = locked_by_gap[label]
    recomputed = (summary["formula_se"], summary["bootstrap_se"], summary["quoted_interval"][0], summary["quoted_interval"][1])
    for value, locked_value in zip(recomputed, LOCKED_FIRST_RUN[label]):
        if abs(value - locked_value) > 0.0006:
            locked_run_reproduced = False
    if abs(block_one_se_by_gap[label] / summary["bootstrap_se"] - 1) > 0.10:
        block_one_matches = False

alternating_acf = autocorrelations(np.tile([1.0, -1.0], 50), 2)    # +1, -1, +1, ... for 100 months
long_series_acf = autocorrelations(long_a, 2)                        # the 100,000-month invented series (phi = 0.2) from Part A
by_hand_autocorrs = [0.2 ** lag for lag in range(1, MAX_LAG + 1)]    # 0.2, 0.04, 0.008, ...

print("\n--- Part B: the real test months ---")
run_checks({
    "Part B: 6 series of 105 test months with 12 lags each, so 72 autocorrelations": (
        test_months == EXPECTED_TEST_MONTHS
        and all(len(series) == test_months for series in monthly_series.values())
        and len(monthly_series) * MAX_LAG == 72
    ),
    "autocorrelations: a series alternating +1, -1 gives -0.99 at lag 1 and +0.98 at lag 2 (by hand)": (
        abs(alternating_acf[0] + 0.99) < 1e-9 and abs(alternating_acf[1] - 0.98) < 1e-9
    ),
    "autocorrelations: a long invented series with phi = 0.2 gives about 0.2 at lag 1 and 0.04 at lag 2": (
        abs(long_series_acf[0] - 0.2) < 0.02 and abs(long_series_acf[1] - 0.04) < 0.02
    ),
    "dependence factor: exactly 1 at block length 1, and 1.3958 at length 6 for autocorrelations 0.2, 0.04, ... (by hand)": (
        dependence_factor(by_hand_autocorrs, 1) == 1 and abs(dependence_factor(by_hand_autocorrs, 6) - 1.3958) < 0.001
    ),
    "the locked first run is reproduced: formula SE, bootstrap SE and quoted interval of all three gaps": locked_run_reproduced,
    "block length 1 on the real months gives the locked bootstrap SE of each gap (within 10%)": block_one_matches,
})
print(f"\n{test_months} test months, {first_month} to {last_month}. Autocorrelations use monthly returns; gaps and SEs are annualised.")

# --- Dependence in the data: autocorrelations of the six series
print(f"\nAutocorrelations, lags 1 to {MAX_LAG}. * marks a value outside the band +/-{band:.3f} (1.96 / sqrt({test_months})).")
print("Columns: 60, 120 and 1/N are the strategies' monthly excess returns; a-b is the monthly return of a minus that of b.")
print(f"{'lag':>4}" + "".join(f"{label:>8} " for label in monthly_series))
for lag in range(1, MAX_LAG + 1):
    row = f"{lag:>4}"
    for label in monthly_series:
        value = acf_by_series[label][lag - 1]
        flag = "*" if abs(value) > band else " "
        row += f"{value:>+8.3f}{flag}"
    print(row)

outside_count = 0
for values in acf_by_series.values():
    for value in values:
        if abs(value) > band:
            outside_count += 1
total_values = len(monthly_series) * MAX_LAG
print(f"Outside the band: {outside_count} of {total_values}. Independent months would put about {0.05 * total_values:.1f} outside.")
print("The six series are closely related (the last three are differences of the first three): treat that count as a rough guide.")

# --- What those autocorrelations imply for the SE of an average
print("\nWhat the autocorrelations imply for the SE of an AVERAGE of each series (factor 1 = no change).")
print("The Sharpe gap also depends on the standard deviations, so this is a guide, not a prediction of the gap SEs.")
print(f"{'':<9}" + "".join(f"{'block length ' + str(length):^32}" for length in BLOCK_LENGTHS_REAL))
print(f"{'series':<9}" + f"{'variance':>10}{'SE':>10}{'eff. months':>12}" * len(BLOCK_LENGTHS_REAL))
for label in monthly_series:
    row = f"{label:<9}"
    for length in BLOCK_LENGTHS_REAL:
        factor = dependence_factor(acf_by_series[label], length)
        row += f"{factor:>10.3f}{np.sqrt(factor):>10.3f}{test_months / factor:>12.1f}"
    print(row)
print(f"variance = 1 + 2 x sum of (1 - lag/L) x autocorrelation over lags below L; SE = its square root; eff. months = {test_months} / variance.")
for length in BLOCK_LENGTHS_REAL:
    wobble = 2 * np.sqrt(sum((1 - lag / length) ** 2 for lag in range(1, length)) / test_months)
    print(f"With no dependence at all, the length-{length} variance would still wobble by about +/-{wobble:.2f} (one standard deviation).")

# --- The three gaps: block rows beside the locked rows
block_results = {}
for label, name_a, name_b in GAP_PAIRS:
    locked = locked_by_gap[label]
    print(f"\n{label}: observed gap {locked['gap']:+.3f}")
    print(
        f"  locked first run:  formula SE {locked['formula_se']:.3f} | bootstrap SE {locked['bootstrap_se']:.3f}"
        f" | quoted interval {annual_interval_text(locked['quoted_interval'])}"
    )
    for length in BLOCK_LENGTHS_REAL:
        se, interval = block_summary(
            returns_by_name[name_a], returns_by_name[name_b], length, REPLICATIONS, np.random.default_rng(SEED)
        )
        block_results[(label, length)] = (se, interval)
        zero_inside = interval[0] <= 0 <= interval[1]
        print(
            f"  block length {length}:    SE {se:.3f} ({se / locked['formula_se']:.3f} x the locked formula SE)"
            f" | interval {annual_interval_text(interval)} | zero inside: {zero_inside}"
        )
print("\nBlock intervals are percentile intervals from 5,000 circular-block resamples (seed 42), not studentised.")
print("They are a sensitivity check beside the locked intervals, not a replacement for them.")

# --- The numbers the five locked predictions refer to
print("\nNumbers the locked predictions refer to (the scoring rules were fixed before this run):")
print(f"1. Autocorrelations outside the band: {outside_count} of {total_values}")
print(f"2. Lag-1 autocorrelation of the 1/N series: {acf_by_series['1/N'][0]:+.3f}")
ratio_parts = []
for label in locked_by_gap:
    ratio = block_results[(label, PRIMARY_BLOCK_LENGTH)][0] / locked_by_gap[label]["formula_se"]
    ratio_parts.append(f"{label} {ratio:.3f}")
print(f"3. Block SE at length {PRIMARY_BLOCK_LENGTH} / locked formula SE: " + "; ".join(ratio_parts))
excluding_zero = 0
for se, (low, high) in block_results.values():
    if low > 0 or high < 0:
        excluding_zero += 1
print(f"4. Block intervals (of {len(block_results)}) that exclude zero: {excluding_zero}")
largest_se = max(block_results[(RULE_GAP, length)][0] for length in BLOCK_LENGTHS_REAL)
print(f"5. Largest block SE for {RULE_GAP} over lengths {BLOCK_LENGTHS_REAL}: {largest_se:.3f} (locked rule threshold: {RULE_THRESHOLD})")

