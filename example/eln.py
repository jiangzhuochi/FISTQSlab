"""
对于ELN而言, 行权价越高, 票据价越低, 具有单调性
绘制横坐标为行权价, 纵坐标为票据价, 曲线具有负斜率并凹向原点
当票据价为98.28%(年化10%收益)时, 用牛顿法定价行权价为94.09%

对于ELN的Delta模拟, 发现和short put的Delta类似
单调递减, 在strike附近gamma最大
不同之处在于ELN是有杠杆的, 杠杆为1/strike
S变小的渐近线为y=1/strike
S变大的渐近线为y=0

由于在下部承担的风险加了杠杆
所以如果把ELN的票息折现值比作期权费, 那么比short put拿到的期权费更高
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


def eln_find_strike():
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
    plt.show()


def eln_delta(
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
    start = 0.5
    stop = 1.5
    num = 300
    relative_St_arr = np.linspace(start, stop, num)
    deltas = pd.DataFrame(
        [
            pd.Series(
                [
                    op.print_and_delta_at(t, np.array([St]), price_only=False).delta
                    for St in relative_St_arr
                ],
                index=relative_St_arr,
            )
            for t in range(0, 65, 4)
        ]
    ).T
    plt.plot(deltas)
    plt.axhline(1 / 0.9404, c="#76da91", ls="--", lw=0.5)
    plt.axhline(0, c="#63b2ee", ls="--", lw=0.5)
    plt.axvline(0.9404, c="#63b2ee", ls="--", lw=0.5)
    plt.yticks(np.append(np.arange(0, 1.2, 0.2), 1 / 0.9404))
    plt.annotate("y = 1 / strike, strike=0.9404", xy=(0.47, 1.07))
    plt.annotate("y = 0", xy=(0.47, 0.007))
    plt.annotate("x = strike, strike=0.9404", xy=(0.92, 0.007), rotation=270)
    plt.xlabel("S")
    plt.ylabel("Delta")
    plt.show()

    relative_St_arr = np.linspace(0.8, 1.2, 200)
    prices = pd.DataFrame(
        [
            pd.Series(
                [
                    op.print_and_delta_at(t, np.array([St])).price
                    for St in relative_St_arr
                ],
                index=relative_St_arr,
            )
            for t in range(0, 65, 4)
        ]
    ).T
    plt.plot(prices)
    plt.xlabel("S")
    plt.ylabel("Price")
    plt.show()


if __name__ == "__main__":

    eln_find_strike()
    eln_delta()
