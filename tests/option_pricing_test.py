import json

import numpy as np
import pandas as pd

from fistqslab.option_pricing.euro_option_bs import (
    euro_option_bs,
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
# df = euro_option_bs_series(**kwargs)
# print(df.index)
class DataFrameJSONEncoder(json.JSONEncoder):
    def default(self, df: pd.DataFrame):
        return df.to_dict()


single_op, series_op = euro_option_bs(**kwargs)
print(single_op)
json_str = json.dumps(series_op, cls=DataFrameJSONEncoder)
di = json.loads(json_str)
print(di["price"])
