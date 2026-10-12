price = [100.00, 101.50, 99.80, 102.30, 103.10, 101.75, 104.20, 103.95, 106.40, 105.80]
def compute_returns(price_list) :
 returns = []
 for t in range(1, len(price_list)):
  returns. append((price_list[t]- price_list[t-1])/price_list[t-1])
 return returns
returns = compute_returns(price)
for r in returns:
 if r > 0:
  print(r,"gain")
 elif r < 0:
  print(r,"loss")
 else : 
  print(r,"flat")
