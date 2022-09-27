import os
import time
from pathlib import Path

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


if __name__ == "__main__":
    rainbow_note_price_2d_single()
    rainbow_note_delta_2d_single()
