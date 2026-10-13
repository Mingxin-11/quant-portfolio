import pandas as pd

JUMP_SIZE = 0.4
OLD_FILE = "data/prices.csv"
NEW_FILE = "data/prices_rolling.csv"
CHECK_MONTH = "2020-03"
TBILL_FILE = "data/tbill_TB3MS_downloaded_2026-10-08.csv"
FIRST_MONTH = "2008-01"
LAST_MONTH = "2026-09"

new_prices = pd.read_csv(NEW_FILE, index_col=0, parse_dates=True)
old_prices = pd.read_csv(OLD_FILE, index_col=0, parse_dates=True)
old_prices = old_prices[new_prices.columns]

# Part A: the new file against the old file, on the dates the old file covers
first_old_day = old_prices.index[0]
last_old_day = old_prices.index[-1]
new_same_dates = new_prices.loc[first_old_day:last_old_day]

print("Part A: new file against old file, on the old file's dates")
print("Same dates in both files:", new_same_dates.index.equals(old_prices.index))

level_ratio = new_same_dates / old_prices
return_gap = (new_same_dates.pct_change() - old_prices.pct_change()).abs()
overlap_report = pd.DataFrame({
    "ratio_smallest": level_ratio.min(),
    "ratio_largest": level_ratio.max(),
    "ratio_spread": level_ratio.max() - level_ratio.min(),
    "return_gap_largest": return_gap.max(),
})
print(overlap_report)

# Part B: daily prices to monthly prices to monthly returns
month_of_each_day = new_prices.index.to_period("M")
monthly_prices = new_prices.groupby(month_of_each_day).last()
monthly_returns = monthly_prices.pct_change().dropna()

print("Part B: monthly tables")
print("Monthly prices:", monthly_prices.shape)
print("Monthly returns:", monthly_returns.shape)
print("First monthly return is for:", monthly_returns.index[0])
print(monthly_returns.head(3))

# Part C: one month checked against its daily returns
daily_returns = new_prices.pct_change()
table_return = monthly_returns.loc[CHECK_MONTH]

month_daily = daily_returns.loc[CHECK_MONTH]
compounded = (1 + month_daily).prod() - 1
plain_sum = month_daily.sum()
print("compounded:", compounded)
print("plain sum:", plain_sum)
print("Largest |table - compounded|:", (table_return - compounded).abs().max())

# Part D: the raw T-bill file, looked at before it is used
tbill_raw = pd.read_csv(TBILL_FILE, index_col=0, parse_dates=True, na_values=".")

print("Part D: the raw T-bill file")
print(tbill_raw.head(3))
print(tbill_raw.tail(3))
print(tbill_raw.dtypes)
print("Missing values:", tbill_raw["TB3MS"].isna().sum())

# Part E: annual percent to monthly decimal, labelled by month, cut to our months
tbill_percent = tbill_raw["TB3MS"].to_period("M").loc[FIRST_MONTH:LAST_MONTH]
monthly_rf = (tbill_percent / 100 / 12).rename("rf_monthly")

print("Part E: the monthly T-bill rate")
print("Months:", len(monthly_rf), "| missing:", monthly_rf.isna().sum())
print("First and last month:", monthly_rf.index[0], monthly_rf.index[-1])
print(pd.DataFrame({"TB3MS_percent_per_year": tbill_percent, "rf_monthly": monthly_rf}).head(3))

excess_returns = monthly_returns.sub(monthly_rf, axis=0)
print("Part F: excess returns")
print("Shape:", excess_returns.shape, "| missing values:", excess_returns.isna().sum().sum())
print(excess_returns.head(3))

# Part G: split-like jumps in the daily prices, whole file (counts only)
split_like = daily_returns.abs() > JUMP_SIZE

print("Part G: days with a daily move beyond", JUMP_SIZE)
print(split_like.sum())

# Part H: save the three monthly tables
suffix = f"{FIRST_MONTH}_to_{LAST_MONTH}"
monthly_returns.to_csv(f"data/monthly_returns_{suffix}.csv", index_label="Month")
monthly_rf.to_csv(f"data/monthly_rf_{suffix}.csv", index_label="Month")
excess_returns.to_csv(f"data/monthly_excess_returns_{suffix}.csv", index_label="Month")
print("Part H: saved three files in data/ ending", suffix)

# Part I: read the saved excess-return file back and compare it with the table in memory
reloaded = pd.read_csv(f"data/monthly_excess_returns_{suffix}.csv", index_col=0, parse_dates=True).to_period("M")

print("Part I: read-back check")
print("Shape:", reloaded.shape, "| same months:", reloaded.index.equals(excess_returns.index))
print("Largest difference from the table in memory:", (reloaded - excess_returns).abs().max().max())