"""
对于ELN而言, 行权价越高, 票据价越低, 具有单调性
绘制横坐标为行权价, 纵坐标为票据价, 曲线具有负斜率并凹向原点
当票据价为98.28%(年化10%收益)时, 用牛顿法定价行权价为94.09%
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fistqslab.option_pricing.eln import ELN2, get_eln_strike_from_issue_price
from fistqslab.option_pricing.util import (
    data_path_to_codes_and_all_S_data,
    get_all_price_path,
)

ROOT = Path(".")
IMG_DIR = ROOT / "img"
DATA_DIR = ROOT / "data"
RESULT_DIR = DATA_DIR / "result"

total_time_start = time.perf_counter()


data180101 = get_all_price_path(DATA_DIR / "路径模拟数据/180101.csv")


codes, all_S_data = data_path_to_codes_and_all_S_data(
    {"180101.csv": DATA_DIR / "路径模拟数据/180101.csv"}
)


def elnv(
    codes=codes,
    all_S_data=all_S_data,
    strike=0.9404,
    issue_price=0.9828,
    T=64,
):
    op = ELN2(
        codes=codes,
        all_S_data=all_S_data,
        strike=strike,
        issue_price=issue_price,
        T=T,
    )
    return op.price


if __name__ == "__main__":

    start = 0.9
    step = 0.005
    end = 1 + step
    strike_arange = np.arange(start, end, step)
    prices = list(
        map(
            lambda strike: elnv(strike=strike),
            strike_arange,
        )
    )
    se = pd.Series(prices, index=strike_arange)
    issue_price = 0.9828
    strike_price = get_eln_strike_from_issue_price(
        all_S_data={
            "180101": data180101,
        },
        issue_price=0.9828,
    )
    print(strike_price)
    se.plot(c="#f89588")
    plt.axvline(strike_price, c="#76da91", ls="--")
    plt.axhline(issue_price, c="#63b2ee", ls="--")
    plt.xlabel("strike_price")
    plt.ylabel("issue_price")
    plt.savefig(IMG_DIR / "eln.png")
    plt.show()
