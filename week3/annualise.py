import pandas as pd
import numpy as np
df = pd.read_csv("data/jpm.csv",skiprows=[1,2])
print(df.columns)
df["Price"] = pd. to_datetime(df["Price"])
df = df. set_index("Price")
df. index. name = "Date"
df["Log Return"] = np. log(1+df["Adj Close"].pct_change())
mean_r = df["Log Return"].mean()
std_r = df["Log Return"].std()
annual_log_return = mean_r * 252
annual_return = np.exp(annual_log_return) - 1
annual_volatility = std_r * np.sqrt(252)
print(annual_log_return)
print(annual_return)
print(annual_volatility)