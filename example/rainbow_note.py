import os
import time
from pathlib import Path
from urllib.parse import parse_qsl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

from fistqslab.option_pricing.rainbow_note import RainbowNote2
from fistqslab.option_pricing.util import data_path_to_codes_real_S0_relative_S

plt.rc(
    "axes",
    prop_cycle=cycler(
        "color",
        [
            "#63b2ee",
            "#76da91",
            "#f8cb7f",
            "#f89588",
            "#7cd6cf",
            "#9192ab",
            "#7898e1",
            "#efa666",
            "#eddd86",
            "#9987ce",
        ],
    ),
)

ROOT = Path(".")
IMG_DIR = ROOT / "img"
DATA_DIR = ROOT / "data"
RESULT_DIR = DATA_DIR / "result"

EXAMPLT_DATA = ROOT / "example" / "data"


total_time_start = time.perf_counter()


codes, real_S0, relative_S = data_path_to_codes_real_S0_relative_S(
    {
        "180101.csv": DATA_DIR / "路径模拟数据/180101.csv",
        # "180201.csv": DATA_DIR / "路径模拟数据/180201.csv",
    }
)


def rainbow_note(
    codes=codes,
    real_S0=real_S0,
    all_relative_S_data=relative_S,
    put_strike=0.9,
    lower_call_strike=1,
    upside_participation=1,
    guaranteed_flat_coupon=6.5e-2 * 183 / 365,
    T=183,
):
    return RainbowNote2(
        codes=codes,
        real_S0=real_S0,
        all_relative_S_data=all_relative_S_data,
        put_strike=put_strike,
        lower_call_strike=lower_call_strike,
        upside_participation=upside_participation,
        guaranteed_flat_coupon=guaranteed_flat_coupon,
        T=T,
    )


def rainbow_note_price_2d_single():

    op = rainbow_note()
    relative_St_arr = np.linspace(0.7, 1.3, 1000)
    tlist = [0, 100, 170, 183]
    prices = pd.DataFrame(
        [
            pd.Series(
                [
                    op.price_and_delta_at(
                        t,
                        np.array([u0st]),
                        underlying=0,
                    ).price
                    for u0st in relative_St_arr
                ],
                index=relative_St_arr,
            )
            for t in tlist
        ]
    ).T
    prices.columns = np.array([f"{t=}" for t in tlist])
    prices.plot()

    plt.xlabel("S")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def rainbow_note_delta_2d_single():

    op = rainbow_note()
    relative_St_arr = np.linspace(0.7, 1.3, 1000)
    tlist = [0, 100, 170, 183]
    prices = pd.DataFrame(
        [
            pd.Series(
                [
                    op.price_and_delta_at(
                        t,
                        np.array([u0st]),
                        underlying=0,
                        price_only=False,
                    ).delta
                    for u0st in relative_St_arr
                ],
                index=relative_St_arr,
            )
            for t in tlist
        ]
    ).T
    prices.columns = np.array([f"{t=}" for t in tlist])
    prices.plot()

    plt.xlabel("S")
    plt.ylabel("Delta")
    plt.legend()
    plt.show()


def delta_hedging(
    npaths: int,
    guaranteed_flat_coupon=6.5e-2 * 183 / 365,
):
    op = rainbow_note(guaranteed_flat_coupon=guaranteed_flat_coupon)

    return_ = []
    for npath in range(0, 20000, 20000 // npaths):
        price_delta_list = []
        S = []
        for i in range(op.T + 1):
            St = np.array([op.relative_S[0, npath, i * 250 // 365]])
            S.append(St[0])
            price_delta_list.append(
                op.price_and_delta_at(i, St, underlying=0, price_only=False)
            )
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
        # 最后一天, 清仓已有股票
        cbs[-1] = cbs[-1] + deltas[-1] * S[-1] - guaranteed_flat_coupon
        # 最后一天, 按情况返还现金
        if S[-1] >= 1:
            cbs[-1] = cbs[-1] - S[-1]
        elif S[-1] < 1 and S[-1] >= 0.9:
            cbs[-1] = cbs[-1] - 1
        else:
            cbs[-1] = cbs[-1] - S[-1] / 0.9
        return_.append(cbs[-1])
        # plt.plot(np.array(cbs))
    # plt.show()

    return_arr = np.array(return_)
    pd.DataFrame(return_arr).to_csv(EXAMPLT_DATA / "rainbow.csv")
    print(np.mean(return_arr))
    plt.hist(return_arr)
    plt.xlabel("relative return")
    plt.ylabel("frequency")
    plt.show()


if __name__ == "__main__":
    # rainbow_note_price_2d_single()
    # rainbow_note_delta_2d_single()
    # delta_hedging(100)
    print(rainbow_note().price())
