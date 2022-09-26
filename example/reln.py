"""
对于RELN而言, 通常固定strike, 给出票面价, 即利息
由于投资者需要以股票作为本金支付, 因此支付了100%
绘制横坐标为票面价, 纵坐标为期权价值
当期权价值为1时(即投资者要想不亏不赚), 牛顿法定价票面价为1.0245
即利息为0.0245, 换成年化为13.97%
如果券商给的利息比年化13.97%低, 那么是盈利的

对RELN的price和delta分析发现
RELN买方相当于卖出虚值看跌期权, 权利金为行权价加利息
但是由于需要支付全额本金, 因此杠杆很低
称之为买入备兑看涨期权(持有标的，并卖出标的相对应的看涨期权)更合适。

两者区别在于：
1) 卖出看跌期权一旦标的价格下跌, 其亏损是无其他头寸对冲, 
而备兑看涨期权只要标的价格不大幅度下跌, 
卖出看涨期权的收益可以在一定程度上对冲标的价格的下跌亏损, 甚至依旧取得收益。
2) 单纯的卖出看跌期权是以标的价格不跌或跌幅有限为前提, 
一般来说标的价格大幅上涨对其是有利的；
而备兑看涨期权是以标的价格不跌或者跌幅有限, 但也不希望标的价格大幅上涨。

日度delta对冲模拟一千条路径, 平均期末损益为0.00012, 即万分之1.2(理想情况为0)
即如果名义本金是100万元, 那么到期平均损益为120元
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


def reln_delta():
    op = reln()
    start = 0.7
    stop = 1.4
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
    plt.axhline(1, c="#76da91", ls="--", lw=0.5)
    plt.axhline(0, c="#63b2ee", ls="--", lw=0.5)
    plt.axvline(1.035, c="#63b2ee", ls="--", lw=0.5)
    plt.annotate("x = strike, strike=1.035", xy=(1.020, 0.007), rotation=270)
    plt.xlabel("S")
    plt.ylabel("Delta")
    plt.show()


def reln_price():
    op = reln()
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
    plt.axhline(1.035 + 1.0172 - 1, c="#76da91", ls="--", lw=0.5)
    plt.annotate(
        "y = strike + issue_price - 1, strike=1.035, issue_price=1.0172",
        xy=(0.79, 1.055),
    )
    plt.xlabel("S")
    plt.ylabel("Price")
    plt.show()


def delta_hedging(
    npaths: int,
    strike=1.0350,
    issue_price=1.0172,
):
    op = reln(
        strike=strike,
        issue_price=issue_price,
    )

    return_ = []
    for npath in range(0, 20000, 20000 // npaths):
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
        # 最后一天, 对方行权, 按情况返还现金, 并清仓已有股票
        if S[-1] > strike:
            cbs[-1] = cbs[-1] - (strike + issue_price - 1) + deltas[-1] * S[-1]
        else:
            cbs[-1] = cbs[-1] - (issue_price - 1) - S[-1] + deltas[-1] * S[-1]
        return_.append(cbs[-1])
        # plt.plot(np.array(cbs))
        # plt.show()

    return_arr = np.array(return_)
    pd.DataFrame(return_arr).to_csv(EXAMPLT_DATA / "reln.csv")
    print(np.mean(return_arr))
    plt.hist(return_arr)
    plt.xlabel("relative return")
    plt.ylabel("frequency")
    plt.show()


if __name__ == "__main__":
    # reln_find_strike()
    # reln_price()
    # reln_delta()
    delta_hedging(1000)
