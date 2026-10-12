import numpy as np
import pandas as pd
dates = ['2024-01-01', '2024-01-02', '2024-01-03']
price = pd. Series([100.00, 101.50, np.nan], index = dates)
print(price)
print(price==np.nan)
print(price.isna())
print(price.isna().sum())