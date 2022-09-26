import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fistqslab.option_pricing.leverage_note import LeverageNote2
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


def leverage_note(
    codes=codes,
    real_S0=real_S0,
    all_relative_S_data=relative_S,
    leverage_multiple=2,
    dividend_rate=5.8e-2,
    leverage_cost=3.95e-2,
    T=365,
):
    return LeverageNote2(
        codes=codes,
        real_S0=real_S0,
        all_relative_S_data=all_relative_S_data,
        leverage_multiple=leverage_multiple,
        dividend_rate=dividend_rate,
        leverage_cost=leverage_cost,
        T=T,
    )


def leverage_note_price():
    op = leverage_note()
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
            for t in [0, 100, 200, 365]
        ]
    ).T
    prices.columns = np.array([f"{t=}" for t in [0, 100, 200, 365]])
    prices.plot()
    plt.annotate(
        "Delta = leverage_multiple = 2",
        xy=(0.79, 1.35),
    )
    plt.xlabel("S")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def leverage_note_delta():
    op = leverage_note()
    start = 0.7
    stop = 1.4
    num = 300
    relative_St_arr = np.linspace(start, stop, num)
    deltas = pd.Series(
        [
            op.price_and_delta_at(t, np.array([1]), price_only=False).delta
            for t in range(0, 366)
        ],
    )

    plt.plot(deltas)
    plt.axvline(365, c="#76da91", ls="--", lw=0.5)
    plt.xlabel("t")
    plt.ylabel("Delta")
    plt.show()


def delta_hedging(
    npaths: int,
    leverage_multiple=2,
    dividend_rate=5.8e-2,
):
    op = leverage_note(
        leverage_multiple=leverage_multiple,
        dividend_rate=dividend_rate,
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
        cbs[-1] = (
            cbs[-1]
            + deltas[-1] * S[-1]
            - ((S[-1] - 1 + dividend_rate) * leverage_multiple + 1)
        )
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
    # leverage_note_price()
    # leverage_note_delta()
    delta_hedging(100)
