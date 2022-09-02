import numpy as np
import pandas as pd

from fistqslab.option_pricing.euro_option_bs import (
    EuropeanCallOptionBS,
    euro_option_bs_series,
)

kwargs = {
    "S": 100,
    "L": 100,
    "T": 30 / 365,
    "r": np.log(1 + 0.02),
    "sigma": 0.2,
    "field": "delta",
}
df = euro_option_bs_series(**kwargs)
print(df.index)
