import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

prices = pd.read_csv("data/prices.csv", parse_dates=["Date"], index_col="Date")
log_returns = np.log(prices / prices.shift(1)).dropna()

std_JPM = log_returns["JPM"].std()
std_PG = log_returns["PG"].std()
cov_pandas = log_returns["JPM"].cov(log_returns["PG"])

corr_manual = cov_pandas / (std_JPM * std_PG)
corr_pandas = log_returns["JPM"].corr(log_returns["PG"])

tolerance = 1e-9
print(f"Part 2 - corr_manual: {corr_manual:.9f}")
print(f"Part 2 - corr_pandas: {corr_pandas:.9f}")
print(f"Part 2 - Manual and Pandas agree: {abs(corr_manual - corr_pandas) < tolerance}")

corr_matrix = log_returns.corr()
print("\nPart 3 - Correlation Matrix:")
print(corr_matrix)
diag_is_one = np.allclose(np.diag(corr_matrix), 1.0)
print(f"Part 3 - All diagonal entries equal 1: {diag_is_one}")

fig, ax = plt.subplots()
im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns)
ax.set_yticklabels(corr_matrix.columns)
fig.colorbar(im)

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center")

plt.title("Correlation of Daily Log Returns, 2019–2023")
plt.show()
