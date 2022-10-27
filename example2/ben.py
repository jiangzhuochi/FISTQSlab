from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from scipy.stats import norm

from fistqslab.option_pricing import eu_bs
from fistqslab.option_pricing.util import data_path_to_codes_real_S0_relative_S

ROOT = Path(".")
EXAMPLT2_IMG = ROOT / "example2" / "img"

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


def ben_pricing(
    S=np.array([0.8, 0.94, 1]),
    put_strike=0.9,
    coupon_barrier=1,
    min_redemption=0.8447,
    bonus_coupon=0.16,
    t=0,
    T_days=365,
    sigma=0.2287,
    r=np.log(1.015),
    greeks=None,
):

    T_years = (T_days - t) / 365

    c3 = eu_bs.EuropeanCallOption(
        S=S, L=1 + bonus_coupon, T_years=T_years, r=r, sigma=sigma
    )
    con = eu_bs.CashOrNothingOption(
        S=S,
        L=coupon_barrier,
        T_years=T_years,
        r=r,
        sigma=sigma,
        K=bonus_coupon,
        option_type=1,
    )

    if min_redemption is None:
        """无最低保本比例的 BEN
        1名义本金BEN = 1/put_strike份欧式认沽期权空头(行权价L=put_strike)
                 + 到期可获得1现金的无风险票据
                 + 欧式认购期权多头(行权价L=1+bonus_coupon)
                 + 现金或无认购期权多头(行权价L=coupon_barrier, 现金K=bonus_coupon)"""
        p = eu_bs.EuropeanPutOption(
            S=S, L=put_strike, T_years=T_years, r=r, sigma=sigma
        )
        if greeks == "delta":
            return -1 / put_strike * p.delta + c3.delta + con.delta
        elif greeks == "gamma":
            raise NotImplemented
        elif greeks == "theta":
            raise NotImplemented
        elif greeks == "vega":
            raise NotImplemented
        else:
            return (
                -1 / put_strike * p.price
                + 1 * np.exp(-r * T_years)
                + c3.price
                + con.price
            )

    else:
        """带有最低保本比例的 BEN
        1名义本金BEN = 1/put_strike份牛市价差期权多头(行权价Lmin=put_strike*min_redemption, Lmax=put_strike)
                 + 到期可获得min_redemption现金的无风险票据
                 + 欧式认购期权多头(行权价L=1+bonus_coupon)
                 + 现金或无认购期权多头(行权价L=coupon_barrier, 现金K=bonus_coupon)"""
        c1 = replace(c3, L=put_strike * min_redemption)
        c2 = replace(c3, L=put_strike)
        if greeks == "delta":
            return 1 / put_strike * (c1.delta - c2.delta) + c3.delta + con.delta
        elif greeks == "gamma":
            raise NotImplemented
        elif greeks == "theta":
            raise NotImplemented
        elif greeks == "vega":
            raise NotImplemented
        else:
            return (
                1 / put_strike * (c1.price - c2.price)
                + min_redemption * np.exp(-r * T_years)
                + c3.price
                + con.price
            )


Sarr = np.linspace(0.7, 1.3, 1002)
t_range = [0, 183, 300, 365]
params = {
    "put_strike": 0.9,
    "min_redemption": 0.8447,
    "coupon_barrier": 1,
    "bonus_coupon": 0.16,
}


def base_example():
    eu_bs.plot_sensitivity(
        ben_pricing,
        Sarr,
        t_range,
        params,
        draw_other_lines="\n".join(
            [
                "min_redemption = func_other_params['min_redemption']",
                "bonus_coupon_add1 = 1 + func_other_params['bonus_coupon']",
                "ax.axhline(bonus_coupon_add1, 0, 0.7, ls='--', lw=1)",
                """ax.annotate(f'1 + bonus_coupon = {bonus_coupon_add1}', xy=(0.68,  bonus_coupon_add1+0.01), rotation=0)""",
                "ax.axhline(min_redemption, 0, 0.1, ls='--', lw=1)",
                """ax.annotate(f'min_redemption = {min_redemption}', xy=(0.68,  min_redemption-0.015), rotation=0)""",
                "ax.axvline(bonus_coupon_add1, 0, 0.64, ls='--', lw=1)",
                """ax.annotate(f'1 + bonus_coupon = {bonus_coupon_add1}', xy=( bonus_coupon_add1+0.01, 0.83), rotation=270)""",
            ]
        ),
    )

    plt.savefig(EXAMPLT2_IMG / "ben_保本_price.png", bbox_inches="tight", dpi=200)
    eu_bs.plot_sensitivity(
        ben_pricing,
        Sarr,
        t_range,
        {**params, "greeks": "delta"},
        draw_other_lines="\n".join(
            [
                "put_strike = func_other_params['put_strike']",
                "ax.axhline(1/put_strike, 0, 0.3, ls='--', lw=1)",
                """ax.annotate(f'1 / put_strike = {round(1/put_strike, 3)}', xy=(0.68,  1 / put_strike+0.02), rotation=0)""",
            ]
        ),
    )
    plt.savefig(EXAMPLT2_IMG / "ben_保本_delta.png", bbox_inches="tight", dpi=200)

    eu_bs.plot_sensitivity(
        ben_pricing, Sarr, t_range, {**params, "min_redemption": None}
    )

    plt.savefig(EXAMPLT2_IMG / "ben_非保本_price.png", bbox_inches="tight", dpi=200)

    eu_bs.plot_sensitivity(
        ben_pricing,
        Sarr,
        t_range,
        {**params, "min_redemption": None, "greeks": "delta"},
    )
    plt.savefig(EXAMPLT2_IMG / "ben_非保本_delta.png", bbox_inches="tight", dpi=200)


def risk_analysis():

    # 改变 put strike

    # eu_bs.plot_sensitivity(
    #     ben_pricing,
    #     Sarr,
    #     t_range,
    #     {**params, "put_strike": 0.88},
    #     draw_other_lines="\n".join(
    #         [
    #             "min_redemption = func_other_params['min_redemption']",
    #             "bonus_coupon_add1 = 1 + func_other_params['bonus_coupon']",
    #             "put_strike = func_other_params['put_strike']",
    #             "ax.axhline(bonus_coupon_add1, 0, 0.7, ls='--', lw=1)",
    #             """ax.annotate(f'1 + bonus_coupon = {bonus_coupon_add1}', xy=(0.68,  bonus_coupon_add1+0.01), rotation=0)""",
    #             "ax.axhline(min_redemption, 0, 0.1, ls='--', lw=1)",
    #             """ax.annotate(f'min_redemption = {min_redemption}', xy=(0.68,  min_redemption-0.015), rotation=0)""",
    #             "ax.axvline(bonus_coupon_add1, 0, 0.64, ls='--', lw=1)",
    #             """ax.annotate(f'1 + bonus_coupon = {bonus_coupon_add1}', xy=( bonus_coupon_add1+0.01, 0.83), rotation=270)""",
    #             "ax.axvline(put_strike, 0, 0.34, ls='--', lw=1)",
    #             """ax.annotate(f'put_strike = {put_strike}', xy=( put_strike+0.01, 0.83), rotation=270)""",
    #         ]
    #     ),
    # )

    # plt.savefig(EXAMPLT2_IMG / "ben_保本put_strike0.88_price.png", bbox_inches="tight", dpi=200)
    # eu_bs.plot_sensitivity(
    #     ben_pricing,
    #     Sarr,
    #     t_range,
    #     {**params, "put_strike": 0.88, "greeks": "delta"},
    #     draw_other_lines="\n".join(
    #         [
    #             "put_strike = func_other_params['put_strike']",
    #             "ax.axhline(1/put_strike, 0, 0.3, ls='--', lw=1)",
    #             """ax.annotate(f'1 / put_strike = {round(1/put_strike, 3)}', xy=(0.68,  1 / put_strike+0.02), rotation=0)""",
    #         ]
    #     ),
    # )
    # plt.savefig(EXAMPLT2_IMG / "ben_保本put_strike0.88_delta.png", bbox_inches="tight", dpi=200)

    # 改变 coupon_barrier
    # eu_bs.plot_sensitivity(
    #     ben_pricing,
    #     Sarr,
    #     t_range,
    #     {**params, "coupon_barrier": 1.1},
    #     draw_other_lines="\n".join(
    #         [
    #             "min_redemption = func_other_params['min_redemption']",
    #             "bonus_coupon_add1 = 1 + func_other_params['bonus_coupon']",
    #             "put_strike = func_other_params['put_strike']",
    #             "ax.axhline(bonus_coupon_add1, 0, 0.7, ls='--', lw=1)",
    #             """ax.annotate(f'1 + bonus_coupon = {bonus_coupon_add1}', xy=(0.68,  bonus_coupon_add1+0.01), rotation=0)""",
    #             "ax.axhline(min_redemption, 0, 0.1, ls='--', lw=1)",
    #             """ax.annotate(f'min_redemption = {min_redemption}', xy=(0.68,  min_redemption-0.015), rotation=0)""",
    #             "ax.axvline(bonus_coupon_add1, 0, 0.64, ls='--', lw=1)",
    #             """ax.annotate(f'1 + bonus_coupon = {bonus_coupon_add1}', xy=( bonus_coupon_add1+0.01, 0.83), rotation=270)""",
    #             "ax.axvline(put_strike, 0, 0.34, ls='--', lw=1)",
    #             """ax.annotate(f'put_strike = {put_strike}', xy=( put_strike+0.01, 0.83), rotation=270)""",
    #         ]
    #     ),
    # )
    # plt.savefig(EXAMPLT2_IMG / "ben_保本coupon_barrier1.1_price.png", bbox_inches="tight", dpi=200)
    # eu_bs.plot_sensitivity(
    #     ben_pricing,
    #     Sarr,
    #     t_range,
    #     {**params, "put_strike": 0.88, "greeks": "delta"},
    #     draw_other_lines="\n".join(
    #         [
    #             "put_strike = func_other_params['put_strike']",
    #             "ax.axhline(1/put_strike, 0, 0.3, ls='--', lw=1)",
    #             """ax.annotate(f'1 / put_strike = {round(1/put_strike, 3)}', xy=(0.68,  1 / put_strike+0.02), rotation=0)""",
    #         ]
    #     ),
    # )

    # plt.savefig(EXAMPLT2_IMG / "ben_保本coupon_barrier1.1_delta.png", bbox_inches="tight", dpi=200)

    # 改变 min_redemption
    # eu_bs.plot_sensitivity(
    #     ben_pricing,
    #     Sarr,
    #     t_range,
    #     {**params, "min_redemption": 0.8},
    #     draw_other_lines="\n".join(
    #         [
    #             "min_redemption = func_other_params['min_redemption']",
    #             "bonus_coupon_add1 = 1 + func_other_params['bonus_coupon']",
    #             "put_strike = func_other_params['put_strike']",
    #             "ax.axhline(bonus_coupon_add1, 0, 0.7, ls='--', lw=1)",
    #             """ax.annotate(f'1 + bonus_coupon = {bonus_coupon_add1}', xy=(0.68,  bonus_coupon_add1+0.01), rotation=0)""",
    #             "ax.axhline(min_redemption, 0, 0.1, ls='--', lw=1)",
    #             """ax.annotate(f'min_redemption = {min_redemption}', xy=(0.68,  min_redemption-0.015), rotation=0)""",
    #             "ax.axvline(bonus_coupon_add1, 0, 0.64, ls='--', lw=1)",
    #             """ax.annotate(f'1 + bonus_coupon = {bonus_coupon_add1}', xy=( bonus_coupon_add1+0.01, 0.83), rotation=270)""",
    #             "ax.axvline(put_strike, 0, 0.34, ls='--', lw=1)",
    #             """ax.annotate(f'put_strike = {put_strike}', xy=( put_strike+0.01, 0.83), rotation=270)""",
    #         ]
    #     ),
    # )
    # plt.savefig(
    #     EXAMPLT2_IMG / "ben_保本min_redemption0.8_price.png", bbox_inches="tight", dpi=200
    # )
    # eu_bs.plot_sensitivity(
    #     ben_pricing,
    #     Sarr,
    #     t_range,
    #     {**params, "min_redemption": 0.8, "greeks": "delta"},
    #     draw_other_lines="\n".join(
    #         [
    #             "put_strike = func_other_params['put_strike']",
    #             "ax.axhline(1/put_strike, 0, 0.3, ls='--', lw=1)",
    #             """ax.annotate(f'1 / put_strike = {round(1/put_strike, 3)}', xy=(0.68,  1 / put_strike+0.02), rotation=0)""",
    #         ]
    #     ),
    # )

    # plt.savefig(
    #     EXAMPLT2_IMG / "ben_保本min_redemption0.8_delta.png", bbox_inches="tight", dpi=200
    # )

    # bonus_coupon
    eu_bs.plot_sensitivity(
        ben_pricing,
        Sarr,
        t_range,
        {**params, "bonus_coupon": 0.2},
        draw_other_lines="\n".join(
            [
                "min_redemption = func_other_params['min_redemption']",
                "bonus_coupon_add1 = 1 + func_other_params['bonus_coupon']",
                "put_strike = func_other_params['put_strike']",
                "ax.axhline(bonus_coupon_add1, 0, 0.7, ls='--', lw=1)",
                """ax.annotate(f'1 + bonus_coupon = {bonus_coupon_add1}', xy=(0.68,  bonus_coupon_add1+0.01), rotation=0)""",
                "ax.axhline(min_redemption, 0, 0.1, ls='--', lw=1)",
                """ax.annotate(f'min_redemption = {min_redemption}', xy=(0.68,  min_redemption-0.015), rotation=0)""",
                "ax.axvline(bonus_coupon_add1, 0, 0.64, ls='--', lw=1)",
                """ax.annotate(f'1 + bonus_coupon = {bonus_coupon_add1}', xy=( bonus_coupon_add1+0.01, 0.83), rotation=270)""",
                "ax.axvline(put_strike, 0, 0.34, ls='--', lw=1)",
                """ax.annotate(f'put_strike = {put_strike}', xy=( put_strike+0.01, 0.83), rotation=270)""",
            ]
        ),
    )
    plt.savefig(
        EXAMPLT2_IMG / "ben_保本bonus_coupon0.2_price.png", bbox_inches="tight", dpi=200
    )
    eu_bs.plot_sensitivity(
        ben_pricing,
        Sarr,
        t_range,
        {**params, "bonus_coupon": 0.2, "greeks": "delta"},
        draw_other_lines="\n".join(
            [
                "put_strike = func_other_params['put_strike']",
                "ax.axhline(1/put_strike, 0, 0.3, ls='--', lw=1)",
                """ax.annotate(f'1 / put_strike = {round(1/put_strike, 3)}', xy=(0.68,  1 / put_strike+0.02), rotation=0)""",
            ]
        ),
    )

    plt.savefig(
        EXAMPLT2_IMG / "ben_保本bonus_coupon0.2_delta.png", bbox_inches="tight", dpi=200
    )


def plot_S_sigma_price():

    eu_bs.plot_sigma_sensitivity(
        ben_pricing,
        Sarr,
        np.arange(0.1, 0.4, 0.1),
        params,
    )
    plt.savefig(EXAMPLT2_IMG / "ben_sigma_price.png", bbox_inches="tight", dpi=200)

    eu_bs.plot_sigma_sensitivity(
        ben_pricing,
        Sarr,
        np.arange(0.1, 0.4, 0.1),
        {**params, "put_strike": 0.88},
    )
    plt.savefig(
        EXAMPLT2_IMG / "ben_sigma_put_strike0.88_price.png",
        bbox_inches="tight",
        dpi=200,
    )

    eu_bs.plot_sigma_sensitivity(
        ben_pricing,
        Sarr,
        np.arange(0.1, 0.4, 0.1),
        {**params, "coupon_barrier": 1.1},
    )
    plt.savefig(
        EXAMPLT2_IMG / "ben_sigma_coupon_barrier1.1_price.png",
        bbox_inches="tight",
        dpi=200,
    )

    eu_bs.plot_sigma_sensitivity(
        ben_pricing,
        Sarr,
        np.arange(0.1, 0.4, 0.1),
        {**params, "min_redemption": 0.8},
    )
    plt.savefig(
        EXAMPLT2_IMG / "ben_sigma_umin_redemption0.8_price.png",
        bbox_inches="tight",
        dpi=200,
    )

    eu_bs.plot_sigma_sensitivity(
        ben_pricing,
        Sarr,
        np.arange(0.1, 0.4, 0.1),
        {**params, "bonus_coupon": 0.2},
    )
    plt.savefig(
        EXAMPLT2_IMG / "ben_sigma_bonus_coupon0.2_price.png",
        bbox_inches="tight",
        dpi=200,
    )


ROOT = Path(".")
DATA_DIR = ROOT / "data"
codes, real_S0, relative_S = data_path_to_codes_real_S0_relative_S(
    {
        "180101.csv": DATA_DIR / "路径模拟数据/180101.csv",
    }
)


def ben_delta_hedging(T=365):

    S = relative_S[0][:, [t * 250 // 365 for t in range(T + 1)]]
    prices = np.empty(S.shape)
    deltas = np.empty(S.shape)
    for idx in range(S.shape[1]):
        prices[:, idx] = ben_pricing(S[:, idx], t=idx, **params)
        deltas[:, idx] = ben_pricing(S[:, idx], t=idx, greeks="delta", **params)
    deltas = np.clip(deltas, -1.2, 1.2)
    delta_chgs = np.diff(deltas, axis=1)
    # 第1天往后, delta变大, 就买股票, 反之就卖
    delta_rehedge_cfs = -delta_chgs * S[:, 1:]
    # 第0天, 卖期权(卖方负delta), 拿现金, 买股票
    cbs = np.empty(S.shape)
    cbs[:, 0] = prices[:, 0] - deltas[:, 0] * S[:, 0]
    for j in range(delta_rehedge_cfs.shape[1]):
        new_cash = cbs[:, j] * (1 + 0.015 / 365)
        cbs[:, j + 1] = new_cash + delta_rehedge_cfs[:, j]

    # 最后一天, 清仓已有股票
    cbs[:, -1] = cbs[:, -1] + deltas[:, -1] * S[:, -1]
    put_strike = 0.9
    coupon_barrier = 1
    min_redemption = 0.8447
    bonus_coupon = 0.16

    def calc_last_cb(cb, s):
        if s >= coupon_barrier:
            return cb - max(s, 1 + bonus_coupon)
        elif put_strike <= s < coupon_barrier:
            return cb - 1
        else:
            return cb - max(min_redemption, s / put_strike)

    # 最后一天, 按情况返还现金
    cbs[:, -1] = list(map(calc_last_cb, cbs[:, -1], S[:, -1]))
    plt.hist(cbs[:, -1], bins=50)
    plt.xlim(-0.1, 0.1)
    plt.savefig(
        EXAMPLT2_IMG / "ben_sigma_umin_redemption0.8_price.png",
        bbox_inches="tight",
        dpi=200,
    )
    print(pd.Series(cbs[:, -1]).describe())

    

# risk_analysis()
# plot_S_sigma_price()
ben_delta_hedging()
