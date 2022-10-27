from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from scipy.stats import norm

from fistqslab.option_pricing import eu_bs

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


def rb_pricing(
    S=np.array([0.8, 0.94, 1]),
    put_strike=0.9,
    lower_call_strike=1,
    upside_participation=1,
    guaranteed_flat_coupon=6.5e-2 * 183 / 365,
    t=0,
    T_days=183,
    sigma=0.228677,
    r=np.log(1.015),
    greeks=None,
):

    T_years = (T_days - t) / 365

    c = eu_bs.EuropeanCallOption(
        S=S, L=lower_call_strike, T_years=T_years, r=r, sigma=sigma
    )
    p = eu_bs.EuropeanPutOption(S=S, L=put_strike, T_years=T_years, r=r, sigma=sigma)

    """RB
    1名义本金RB = 1/put_strike份欧式认沽期权空头(行权价L=put_strike)
             + 到期可获得1+guaranteed_flat_coupon现金的无风险票据
             + upside_participation份欧式认购期权多头(行权价L=lower_call_strike)"""

    if greeks == "delta":
        return upside_participation * c.delta - (1 / put_strike) * p.delta
    elif greeks == "gamma":
        raise NotImplemented
    elif greeks == "theta":
        raise NotImplemented
    elif greeks == "vega":
        raise NotImplemented
    else:
        return (
            upside_participation * c.price
            - (1 / put_strike) * p.price
            + (1 + guaranteed_flat_coupon) * np.exp(-r * T_years)
        )


Sarr = np.linspace(0.7, 1.3, 1002)
t_range = [0, 100, 170, 183]
params = {
    "put_strike": 0.9,
    "lower_call_strike": 1,
    "guaranteed_flat_coupon": 6.5e-2 * 183 / 365,
    "upside_participation": 1,
}


def plot_S_t_price_delta():
    eu_bs.plot_sensitivity(
        rb_pricing,
        Sarr,
        t_range,
        params,
        draw_other_lines="\n".join(
            [
                "guaranteed_flat_coupon = func_other_params['guaranteed_flat_coupon']",
                "ax.axhline(1 + guaranteed_flat_coupon, 0, 0.4, ls='--', lw=1)",
                """ax.annotate(f'1 + guaranteed_flat_coupon = {round(1 + guaranteed_flat_coupon,3)}', xy=(0.68,  1 + guaranteed_flat_coupon+0.03), rotation=0)""",
            ]
        ),
    )
    plt.savefig(EXAMPLT2_IMG / "rb_par1_price.png", bbox_inches="tight", dpi=200)

    eu_bs.plot_sensitivity(
        rb_pricing,
        Sarr,
        t_range,
        {**params, "greeks": "delta"},
        draw_other_lines="\n".join(
            [
                "upside_participation = func_other_params['upside_participation']",
                "put_strike = func_other_params['put_strike']",
                "ax.axhline(1/put_strike, 0, 0.3, ls='--', lw=1)",
                """ax.annotate(f'1 / put_strike = {round(1 / put_strike, 3)}, put_strike = {put_strike}', xy=(0.68,  1 / put_strike+0.02), rotation=0)""",
            ]
        ),
    )
    plt.savefig(EXAMPLT2_IMG / "rb_par1_delta.png", bbox_inches="tight", dpi=200)

    # 改变 upside_participation price

    eu_bs.plot_sensitivity(
        rb_pricing,
        Sarr,
        t_range,
        {**params, "upside_participation": 1.5},
        draw_other_lines="\n".join(
            [
                "guaranteed_flat_coupon = func_other_params['guaranteed_flat_coupon']",
                "ax.axhline(1 + guaranteed_flat_coupon, 0, 0.4, ls='--', lw=1)",
                """ax.annotate(f'1 + guaranteed_flat_coupon = {round(1 + guaranteed_flat_coupon,3)}', xy=(0.68,  1 + guaranteed_flat_coupon+0.03), rotation=0)""",
            ]
        ),
    )
    plt.savefig(EXAMPLT2_IMG / "rb_par1.5_price.png", bbox_inches="tight", dpi=200)

    # 改变 upside_participation delta

    eu_bs.plot_sensitivity(
        rb_pricing,
        Sarr,
        t_range,
        {**params, "upside_participation": 1.5, "greeks": "delta"},
        draw_other_lines="\n".join(
            [
                "upside_participation = func_other_params['upside_participation']",
                "put_strike = func_other_params['put_strike']",
                "ax.axhline(upside_participation, 0, 0.6, ls='--', lw=1)",
                """ax.annotate(f'upside_participation = {upside_participation}', xy=(0.68,  upside_participation+0.03), rotation=0)""",
                "ax.axhline(1/put_strike, 0, 0.3, ls='--', lw=1)",
                """ax.annotate(f'1 / put_strike = {round(1 / put_strike, 3)}, put_strike = {put_strike}', xy=(0.68,  1 / put_strike+0.03), rotation=0)""",
            ]
        ),
    )
    plt.savefig(EXAMPLT2_IMG / "rb_par1.5_delta.png", bbox_inches="tight", dpi=200)

    # 改变 put_strike price

    eu_bs.plot_sensitivity(
        rb_pricing,
        Sarr,
        t_range,
        {**params, "put_strike": 0.8},
        draw_other_lines="\n".join(
            [
                "guaranteed_flat_coupon = func_other_params['guaranteed_flat_coupon']",
                "ax.axhline(1 + guaranteed_flat_coupon, 0, 0.4, ls='--', lw=1)",
                """ax.annotate(f'1 + guaranteed_flat_coupon = {round(1 + guaranteed_flat_coupon,3)}', xy=(0.68,  1 + guaranteed_flat_coupon+0.03), rotation=0)""",
            ]
        ),
    )
    plt.savefig(
        EXAMPLT2_IMG / "rb_put_strike0.8_price.png", bbox_inches="tight", dpi=200
    )

    # 改变 put_strike delta

    eu_bs.plot_sensitivity(
        rb_pricing,
        Sarr,
        t_range,
        {**params, "put_strike": 0.8, "greeks": "delta"},
        draw_other_lines="\n".join(
            [
                "upside_participation = func_other_params['upside_participation']",
                "put_strike = func_other_params['put_strike']",
                "ax.axhline(upside_participation, 0, 0.6, ls='--', lw=1)",
                """ax.annotate(f'upside_participation = {upside_participation}', xy=(0.68,  upside_participation+0.03), rotation=0)""",
                "ax.axhline(1/put_strike, 0, 0.3, ls='--', lw=1)",
                """ax.annotate(f'1 / put_strike = {round(1 / put_strike, 3)}, put_strike = {put_strike}', xy=(0.68,  1 / put_strike+0.03), rotation=0)""",
            ]
        ),
    )
    plt.savefig(
        EXAMPLT2_IMG / "rb_put_strike0.8_delta.png", bbox_inches="tight", dpi=200
    )


def plot_S_sigma_price():

    eu_bs.plot_sigma_sensitivity(
        rb_pricing,
        Sarr,
        np.arange(0.1, 0.4, 0.1),
        params,
    )
    plt.savefig(EXAMPLT2_IMG / "rb_sigma_price.png", bbox_inches="tight", dpi=200)


    eu_bs.plot_sigma_sensitivity(
        rb_pricing,
        Sarr,
        np.arange(0.1, 0.4, 0.1),
        {**params, "lower_call_strike": 1.2},
    )
    plt.savefig(
        EXAMPLT2_IMG / "rb_sigma_lower_call_strike1.2_price.png",
        bbox_inches="tight",
        dpi=200,
    )


    eu_bs.plot_sigma_sensitivity(
        rb_pricing,
        Sarr,
        np.arange(0.1, 0.4, 0.1),
        {**params, "put_strike": 0.8},
    )
    plt.savefig(
        EXAMPLT2_IMG / "rb_sigma_put_strike0.8_price.png", bbox_inches="tight", dpi=200
    )


    eu_bs.plot_sigma_sensitivity(
        rb_pricing,
        Sarr,
        np.arange(0.1, 0.4, 0.1),
        {**params, "upside_participation": 1.5},
    )
    plt.savefig(
        EXAMPLT2_IMG / "rb_sigma_upside_participation1.5_price.png",
        bbox_inches="tight",
        dpi=200,
    )

# plot_S_sigma_price()



d = rb_pricing()
print(d)
