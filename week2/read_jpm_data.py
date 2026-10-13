import pandas as pd
price = pd.read_csv("data/jpm.csv",skiprows=[1,2])
print(price. head())
print(price. dtypes)
print(price["Close"].mean())
print(price.shape)
print(price.columns)
price.info()
price["ratio"] = price["Close"] / price["Adj Close"]
print(price.head())
print(price.tail())
close_return = (price["Close"].iloc[1] / price["Close"].iloc[0]) - 1
adj_close_return = (price["Adj Close"].iloc[1] / price["Adj Close"].iloc[0]) - 1
print(["Close Return"])
print(["adj_close return"])
print(price.isna().sum())
price["return"] = price["Adj Close"].pct_change()
print(price.head())