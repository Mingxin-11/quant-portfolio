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
std_r_pop = df["Log Return"].std(ddof=0)
print(df["Log Return"].describe())
print(mean_r)
print(std_r)
print(std_r_pop)