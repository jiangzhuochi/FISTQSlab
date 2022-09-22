import numpy as np
import pandas as pd


def get_data():
    """示例ohlc和sig数据"""
    df = pd.read_csv("data/ohlc.csv", index_col=0)
    num_of_rows = df.shape[0]
    df["sig"] = np.random.randint(-1, 2, num_of_rows)
    df.reset_index(inplace=True)
    df = df.reindex(columns=["Date", "open", "close", "low", "high", "sig"])
    # print(df.to_numpy())
    return df.to_numpy().tolist()
