import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("data/jpm.csv", skiprows=[1,2])
df["Price"] = pd.to_datetime(df["Price"])
df = df.set_index("Price")
df.index.name = "Date"

df["Daily Return"] = df["Adj Close"].pct_change()
df["Log Return"] = np.log(1+ df["Daily Return"])
mean_r = df["Log Return"].mean()
std_r = df["Log Return"].std()

def annualise (daily_mean, daily_std):
    annual_log_return = daily_mean * 252
    annual_return = np.exp(annual_log_return) - 1
    annual_volatility = daily_std * np.sqrt(252)
    return annual_return, annual_volatility

annual_return, annual_volatility = annualise(mean_r, std_r)
print(mean_r)
print(std_r)
print(annual_volatility)
print(annual_return)

worst_log_return = df["Log Return"].min()
worst_z_score = (worst_log_return - mean_r) / std_r
print(worst_log_return)
print(worst_z_score)

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["Adj Close"], label="JPM Adjusted Close Price", color="navy")
plt.title("JPM Adjusted Close Price (5-Year History)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["Log Return"], label="Daily Log Return", color="teal", alpha=0.7)
plt.title("JPM Daily Log Returns")
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()

log_returns = df["Log Return"].dropna()
plt.figure(figsize=(10, 5))
plt.hist(log_returns, bins=50, density=True, color="steelblue", edgecolor="white", alpha=0.8)
x = np.linspace(log_returns.min(), log_returns.max(), 500)
normal_curve = (1 / (std_r * np.sqrt(2 * np.pi))) * np.exp(-((x - mean_r) ** 2) / (2 * std_r ** 2))
plt.plot(x, normal_curve, color="crimson", linewidth=2, label="Normal Curve")
plt.title("JPM Daily Log Returns vs. Normal Distribution")
plt.xlabel("Daily Log Return")
plt.ylabel("Density")
plt.legend()
plt.show()