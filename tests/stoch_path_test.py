import timeit

import matplotlib.pyplot as plt

from fistqslab.option_pricing.util import get_stock_path_1y

al = get_stock_path_1y(S0=100, mu=0.02, vol=0.2, m_step=50, n_path=10)
plt.plot(al)
# plt.show()

time_costs = timeit.timeit(
    "get_stock_path_1y(S0=100, mu=0.02, vol=0.2)",
    "from fistqslab.option_pricing.util import get_stock_path_1y",
    number=1,
)
print(time_costs)
