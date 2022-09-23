"""
在投资者视角, 对于ELN而言, 行权价越高, 票据价越低, 具有单调性
绘制横坐标为行权价, 纵坐标为票据价, 曲线具有负斜率并凹向原点
当票据价为98.28%(年化10%收益)时, 用牛顿法定价行权价为94.09%

对于ELN的delta模拟, 发现和short put的delta类似
单调递减, gamma为负, 在strike附近gamma绝对值最大
不同之处在于ELN是有杠杆的, 杠杆为1/strike
S变小的渐近线为y=1/strike
S变大的渐近线为y=0

由于在下部承担的风险加了杠杆
所以如果把ELN的票息折现值比作期权费, 那么比short put拿到的期权费更高

由于券商卖出ELN获得正gamma
在此期间券商delta对冲是高抛低吸
赚取的对冲收益用来支付对投资者承诺的票息

日度delta对冲模拟一千条路径, 平均期末损益为0.00016, 即万分之1.6(理想情况为0)
即如果名义本金是100万元, 那么到期平均损益为160元, 最大损益不超过2%.

如果券商盈利, 则真实strike比牛顿法定价的strike要更高
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

EXAMPLT_DATA = ROOT / "example" / "data"

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
    return op


def eln_find_strike():
    start = 0.9
    stop = 1
    num = 100
    strike_arange = np.linspace(start, stop, num)

    prices = list(
        map(
            lambda strike: elnv(strike=strike).price(),
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
                    op.price_and_delta_at(t, np.array([St]), price_only=False).delta
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
                    op.price_and_delta_at(t, np.array([St])).price
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


def delta_hedging():
    op = elnv()
    # print(op)
    # print(op.price())
    # 先取1条
    sample_delta_hedging_paths = op.relative_S[0, 0, :]
    # print(sample_delta_hedging_paths)

    return_ = []
    for npath in range(0, 20000, 20):
        price_delta_list = []
        S = []
        for i in range(op.T + 1):
            St = np.array([op.relative_S[0, npath, i * 250 // 365]])
            S.append(St[0])
            price_delta_list.append(op.price_and_delta_at(i, St, price_only=False))
        price_delta = np.array(price_delta_list)
        prices = price_delta[:, 0]
        deltas = price_delta[:, 1]
        delta_chgs = np.diff(deltas)
        # 第1天往后, delta变大, 就买股票, 反之就卖(对卖方而言, 相当于高抛低吸)
        delta_rehedge_cfs = -delta_chgs * S[1:]
        # 第0天, 卖期权(卖方负delta), 拿现金, 买股票
        start_cbs = prices[0] - deltas[0] * S[0]
        cbs = [start_cbs]
        for cf in delta_rehedge_cfs:
            prev_cash = cbs[-1]
            new_cash = prev_cash * (1 + 0.015 / 365)
            cb = new_cash + cf
            cbs.append(cb)
        # 最后一天, 对方行权, 按情况返还现金
        if S[-1] > 0.9404:
            cbs[-1] = cbs[-1] - 1 + deltas[-1] * S[-1]
        else:
            cbs[-1] = cbs[-1] - 1 / 0.9404 * S[-1] + deltas[-1] * S[-1]
        return_.append(cbs[-1])
        # plt.plot(np.array(cbs))
        # plt.show()

    return_arr = np.array(return_)
    pd.DataFrame(return_arr).to_csv(EXAMPLT_DATA / "eln.csv")
    print(np.mean(return_arr))
    plt.hist(return_arr)
    plt.xlabel("relative return")
    plt.ylabel("frequency")
    plt.show()


if __name__ == "__main__":

    # eln_find_strike()
    # eln_delta()
    delta_hedging()
