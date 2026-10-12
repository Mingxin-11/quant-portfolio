price = [50.00, 52.00, 49.40, 53.35, 53.35]
returns = []
for t in range(1, len(price)):
 returns. append((price[t]- price[t-1])/price[t-1])
print(returns)
print(t)