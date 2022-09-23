"""
对于RELN而言, 通常固定strike, 给出票面价, 即利息
由于投资者需要以股票作为本金支付, 因此支付了100%
绘制横坐标为票面价, 纵坐标为期权价值
当期权价值为1时(即投资者要想不亏不赚), 牛顿法定价票面价为1.0245
即利息为0.0245, 换成年化为13.97%
如果券商给的利息比年化13.97%低, 那么是盈利的
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fistqslab.option_pricing.eln import RELN2, get_reln_issue_price
from fistqslab.option_pricing.util import data_path_to_codes_real_S0_relative_S

ROOT = Path(".")
IMG_DIR = ROOT / "img"
DATA_DIR = ROOT / "data"
RESULT_DIR = DATA_DIR / "result"

EXAMPLT_DATA = ROOT / "example" / "data"


total_time_start = time.perf_counter()


codes, real_S0, relative_S = data_path_to_codes_real_S0_relative_S(
    {"180101.csv": DATA_DIR / "路径模拟数据/180101.csv"}
)


def reln(
    codes=codes,
    real_S0=real_S0,
    relative_S=relative_S,
    strike=1.0350,
    issue_price=1.0172,
    T=64,
):

    rop = RELN2(
        codes=codes,
        real_S0=real_S0,
        all_relative_S_data=relative_S,
        strike=strike,
        issue_price=issue_price,
        T=T,
    )
    return rop


def reln_find_strike():
    start = 1
    stop = 1.05
    num = 100
    issue_arange = np.linspace(start, stop, num)

    prices = list(
        map(
            lambda issue: reln(issue_price=issue).price(),
            issue_arange,
        )
    )
    se = pd.Series(prices, index=issue_arange)
    issue_price = get_reln_issue_price(codes, real_S0, relative_S, strike=1.0350, T=64)
    plt.axhline(1, c="#76da91", ls="--")
    print(issue_price)
    ticks = np.append(np.arange(1.0, 1.05, 0.01), issue_price)
    plt.xticks(ticks, [str(i) for i in ticks])

    plt.axvline(issue_price, c="#63b2ee", ls="--")
    se.plot(c="#f89588")
    plt.xlabel("issue_price")
    plt.ylabel("option_price")
    plt.show()


reln_find_strike()
