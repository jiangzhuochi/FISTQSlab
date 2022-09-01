import numpy as np
import pandas as pd

from fistqslab.option_pricing import vanilla_price_series

args = (100, 100, 30 / 365, np.log(1 + 0.02), 0.2, "delta")
df = vanilla_price_series(*args)
print(df.iloc[200, :])
