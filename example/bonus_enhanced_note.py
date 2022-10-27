import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib import cm

from fistqslab.option_pricing.bonus_enhanced_note import BaseBEN2
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
np.set_printoptions(precision=5, edgeitems=2)


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


def bonus_enhanced_note(
    codes=codes,
    real_S0=real_S0,
    all_relative_S_data=relative_S,
    put_strike=0.9,
    coupon_barrier=1,
    bonus_coupon=0.16,
    min_redemption=0.8447,
    T=365,
):
    return BaseBEN2(
        codes=codes,
        real_S0=real_S0,
        all_relative_S_data=all_relative_S_data,
        put_strike=put_strike,
        coupon_barrier=coupon_barrier,
        bonus_coupon=bonus_coupon,
        min_redemption=min_redemption,
        T=T,
    )


def bonus_enhanced_note_price_3d(min_redemption=0.8447):

    op = bonus_enhanced_note(min_redemption=min_redemption)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 8), subplot_kw={"projection": "3d"}
    )

    x_num = 40
    y_num = 40
    u0_relative_St_arr = np.linspace(0.6, 1.2, x_num)
    u1_relative_St_arr = np.linspace(0.6, 1.2, y_num)
    X, Y = np.meshgrid(u0_relative_St_arr, u1_relative_St_arr)
    prices = np.array(
        [
            op.price_and_delta_at(0, np.array(st), underlying=0).price
            for st in zip(X.flat, Y.flat)
        ]
    ).reshape(y_num, x_num)
    ax1.plot_wireframe(  # type:ignore
        X,
        Y,
        np.tile(min_redemption, [x_num, y_num]),
        rstride=10,
        cstride=10,
        cmap=cm.get_cmap("CMRmap"),
    )
    ax1.plot_surface(  # type:ignore
        X, Y, prices, rstride=1, cstride=1, cmap=cm.get_cmap("CMRmap")
    )

    x_num = 60
    y_num = 60
    u0_relative_St_arr = np.linspace(0.6, 1.2, x_num)
    u1_relative_St_arr = np.linspace(0.6, 1.2, y_num)
    X, Y = np.meshgrid(u0_relative_St_arr, u1_relative_St_arr)
    prices = np.array(
        [
            op.price_and_delta_at(365, np.array(st), underlying=0).price
            for st in zip(X.flat, Y.flat)
        ]
    ).reshape(y_num, x_num)
    ax2.plot_wireframe(  # type:ignore
        X,
        Y,
        np.tile(min_redemption, [x_num, y_num]),
        rstride=10,
        cstride=10,
        cmap=cm.get_cmap("CMRmap"),
    )
    ax2.plot_surface(  # type:ignore
        X, Y, prices, rstride=1, cstride=1, cmap=cm.get_cmap("CMRmap")
    )
    ax2.set_zlabel("Price")  # type:ignore

    plt.show()


def bonus_enhanced_note_price_2dst(min_redemption=0.8447):

    op = bonus_enhanced_note(min_redemption=min_redemption)
    relative_St_arr = np.linspace(0.7, 1.3, 300)
    u1st_list = [0.75, 0.85, 0.95, 1.05, 1.15, 1.25]
    prices = pd.DataFrame(
        [
            pd.Series(
                [
                    op.price_and_delta_at(
                        365,
                        np.array([u0st, u1st]),
                        underlying=0,  # 求价格和这个参数无关
                    ).price
                    for u0st in relative_St_arr
                ],
                index=relative_St_arr,
            )
            for u1st in u1st_list
        ]
    ).T
    prices.columns = np.array([f"{u1st=}" for u1st in u1st_list])
    prices.plot()
    plt.axhline(min_redemption, c="#76da91", ls="--", lw=0.5)
    plt.xlabel("u0st")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def bonus_enhanced_note_delta(min_redemption=0.8447):

    op = bonus_enhanced_note(min_redemption=min_redemption)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 8), subplot_kw={"projection": "3d"}
    )

    x_num = 30
    y_num = 30
    u0_relative_St_arr = np.linspace(0.6, 1.2, x_num)
    u1_relative_St_arr = np.linspace(0.6, 1.2, y_num)
    X, Y = np.meshgrid(u0_relative_St_arr, u1_relative_St_arr)
    prices = np.array(
        [
            op.price_and_delta_at(0, np.array(st), underlying=0, price_only=False).delta
            for st in zip(X.flat, Y.flat)
        ]
    ).reshape(y_num, x_num)
    ax1.plot_surface(  # type:ignore
        X, Y, prices, rstride=1, cstride=1, cmap=cm.get_cmap("CMRmap")
    )

    x_num = 60
    y_num = 60
    u0_relative_St_arr = np.linspace(0.6, 1.2, x_num)
    u1_relative_St_arr = np.linspace(0.6, 1.2, y_num)
    X, Y = np.meshgrid(u0_relative_St_arr, u1_relative_St_arr)
    prices = np.array(
        [
            op.price_and_delta_at(
                365, np.array(st), underlying=0, price_only=False
            ).delta
            for st in zip(X.flat, Y.flat)
        ]
    ).reshape(y_num, x_num)

    ax2.plot_surface(  # type:ignore
        X, Y, prices, rstride=1, cstride=1, cmap=cm.get_cmap("CMRmap")
    )
    ax2.set_zlabel("Delta")  # type:ignore

    plt.show()


def bonus_enhanced_note_delta_2ds0(min_redemption=0.8447):

    op = bonus_enhanced_note(min_redemption=min_redemption)
    relative_St_arr = np.linspace(0.7, 1.3, 150)
    u1st_list = [0.75, 0.85, 0.95, 1.05, 1.15, 1.25]
    prices = pd.DataFrame(
        [
            pd.Series(
                [
                    op.price_and_delta_at(
                        0,
                        np.array([u0st, u1st]),
                        underlying=0,
                        price_only=False,
                    ).delta
                    for u0st in relative_St_arr
                ],
                index=relative_St_arr,
            )
            for u1st in [0.75, 0.85, 0.95, 1.05, 1.15, 1.25]
        ]
    ).T
    prices.columns = np.array([f"{u1st=}" for u1st in u1st_list])
    prices = prices.rolling(window=7, center=True).mean().dropna()
    prices.plot()
    plt.axhline(min_redemption, c="#76da91", ls="--", lw=0.5)

    plt.xlabel("u0st")
    plt.ylabel("Delta")
    plt.show()


def bonus_enhanced_note_price_2d_single(min_redemption=0.8447, bonus_coupon=0.16):

    op = bonus_enhanced_note(min_redemption=min_redemption, bonus_coupon=bonus_coupon)
    relative_St_arr = np.linspace(0.7, 1.3, 1000)
    tlist = [0, 183, 360, 365]
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

    plt.axhline(min_redemption, c="#9987ce", ls="--", lw=1)
    plt.axhline(1 + bonus_coupon, c="#eddd86", ls="--", lw=1)
    plt.axhline(1, c="#efa666", ls="--", lw=1)

    plt.xlabel("S")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def bonus_enhanced_note_delta_2d_single(
    put_strike=0.9, min_redemption=0.8447, bonus_coupon=0.16
):

    op = bonus_enhanced_note(
        put_strike=put_strike, min_redemption=min_redemption, bonus_coupon=bonus_coupon
    )
    relative_St_arr = np.linspace(0.7, 1.3, 300)
    # tlist = list(range(361, 366, 2))
    tlist = [0, 183, 300, 365]
    # 183, 300, 365

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
    prices = prices.rolling(window=7, center=True).mean().dropna()
    prices.plot()
    plt.axhline(1, c="#76da91", ls="--", lw=1)
    plt.axhline(1 / put_strike, c="#efa666", ls="--", lw=1)
    plt.ylim(bottom=-0.25, top=1.3)
    plt.xlabel("S")
    plt.ylabel("Delta")
    plt.show()


if __name__ == "__main__":
    # bonus_enhanced_note_price_3d()
    # bonus_enhanced_note_price_2dst()

    # bonus_enhanced_note_delta()
    # bonus_enhanced_note_delta_2ds0()
    # bonus_enhanced_note_price_2d_single()
    # bonus_enhanced_note_delta_2d_single()
    op = bonus_enhanced_note()
    d = op.price_and_delta_at(
        0,
        np.array([1]),
        underlying=0,
        price_only=False,
    )
    print(d)
