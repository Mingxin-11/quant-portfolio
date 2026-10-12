import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("data/jpm.csv", skiprows=[1,2])
df["Price"] = pd.to_datetime(df["Price"])
df = df.set_index("Price")
df.index.name = "Date"
df["Log Return"] = np.log(1 + df["Adj Close"].pct_change())
mean_r = df["Log Return"].mean()
std_r = df["Log Return"].std()

worst_log_return = df["Log Return"].min()
worst_z_score = (worst_log_return - mean_r) / std_r
print(worst_log_return)
print(worst_z_score)

log_returns = df["Log Return"].dropna()
plt.hist(log_returns, bins=50, density=True, color="steelblue", edgecolor="white")
x = np.linspace(log_returns.min(), log_returns.max(), 500)
normal_curve = (1/(std_r*np.sqrt(2*np.pi))) * np.exp(-(x-mean_r)**2/(2*std_r**2))
plt.plot(x, normal_curve, color="crimson", linewidth=2)
plt.xlabel("Daily Log Return")
plt.ylabel("Density")
plt.title("JPM Daily Log Returns vs. Normal Distribution")
plt.show()