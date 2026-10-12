import pandas as pd
dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05','2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10']
price = pd. Series([100.00, 101.50, 99.80, 102.30, 103.10, 101.75, 104.20, 103.95, 106.40, 105.80], index=dates )
print(price.iloc[0],price. loc['2024-01-01'], price. iloc[-1], price. loc['2024-01-10'])
sorted_price = price.sort_values()
print(sorted_price)

