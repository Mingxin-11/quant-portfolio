import pandas as pd
import numpy as np
df = pd.read_csv("data/jpm.csv",skiprows=[1,2])
print(df.columns)
df["Price"] = pd. to_datetime(df["Price"])
df = df. set_index("Price")
df. index. name = "Date"
df["Log Return"] = np. log(1+df["Adj Close"].pct_change())
summed_log_return = df["Log Return"].sum()
p_first = df["Adj Close"].iloc[0]
p_last = df["Adj Close"].iloc[-1]
direct_log_return = np.log(p_last / p_first)
tolerance = 1e-9
difference = abs(summed_log_return - direct_log_return)
is_additive = difference < tolerance
print(summed_log_return)
print(direct_log_return)
print(difference)
print(is_additive)