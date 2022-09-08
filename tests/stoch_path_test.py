import timeit

import matplotlib.pyplot as plt
import numpy as np

from fistqslab.option_pricing.util import get_stock_path_1y

al = get_stock_path_1y(S0=100, mu=0.02 / 250, vol=0.2 / np.sqrt(250), n_path=5)
plt.plot(al[::240, :])
plt.show()

# time_costs = timeit.timeit(
#     "get_stock_path_1y(S0=100, mu=0.02, vol=0.2)",
#     "from fistqslab.option_pricing.util import get_stock_path_1y",
#     number=2,
# )
# print(time_costs)
