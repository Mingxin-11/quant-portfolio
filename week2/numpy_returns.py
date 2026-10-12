import numpy as np
price = np. array([100.00, 101.50, 99.80, 102.30, 103.10, 101.75, 104.20, 103.95, 106.40, 105.80])
price_today = price[1:]
price_yesterday = price[:-1]
returns = price_today / price_yesterday - 1
print(returns)