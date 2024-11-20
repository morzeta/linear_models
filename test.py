from preparation import read_data
from utils import set_random_seed

set_random_seed(1)
x, y = read_data()

print(x.describe())
print(y.head())
