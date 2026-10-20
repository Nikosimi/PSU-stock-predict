from fetch_data import fetch_data
from indicators import add_all_ind

data = fetch_data("AAPL")
data = add_all_ind(data)

print(data)