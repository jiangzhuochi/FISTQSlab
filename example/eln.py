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
from fistqslab.option_pricing.util import data_path_to_codes_real_S0_relative_S

ROOT = Path(".")
IMG_DIR = ROOT / "img"
DATA_DIR = ROOT / "data"
RESULT_DIR = DATA_DIR / "result"

total_time_start = time.perf_counter()


codes, real_S0, relative_S = data_path_to_codes_real_S0_relative_S(
    {"180101.csv": DATA_DIR / "路径模拟数据/180101.csv"}
)


def elnv(
    codes=codes,
    real_S0=real_S0,
    relative_S=relative_S,
    strike=0.9404,
    issue_price=0.9828,
    T=64,
):
    op = ELN2(
        codes=codes,
        real_S0=real_S0,
        all_relative_S_data=relative_S,
        strike=strike,
        issue_price=issue_price,
        T=T,
    )
    return op.price()


if __name__ == "__main__":

    start = 0.9
    stop = 1
    num = 100
    strike_arange = np.linspace(start, stop, num)

    prices = list(
        map(
            lambda strike: elnv(strike=strike),
            strike_arange,
        )
    )
    se = pd.Series(prices, index=strike_arange)
    issue_price = 0.9828
    strike_price = get_eln_strike_from_issue_price(
        codes, real_S0, relative_S, issue_price=0.9828, T=64
    )
    se.plot(c="#f89588")
    plt.axvline(strike_price, c="#76da91", ls="--")
    plt.axhline(issue_price, c="#63b2ee", ls="--")
    plt.xlabel("strike_price")
    plt.ylabel("issue_price")
    plt.savefig(IMG_DIR / "eln.png")
    plt.show()
