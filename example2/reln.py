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


def reln_pricing(
    S=np.array([0.8, 0.94, 1]),
    strike=1.0350,
    issue_price=1.0172,
    t=0,
    T_days=64,
    sigma=0.228677,
    r=np.log(1.015),
    greeks=None,
):
    """RELN 定价

    以1买入1份RELN = 持有到期支付strike+issue_price-1的票据 + 1份欧式认沽期权空头"""

    T_years = (T_days - t) / 365
    p = eu_bs.EuropeanPutOption(S=S, L=strike, T_years=T_years, r=r, sigma=sigma,b=0)
    if greeks == "delta":
        return -p.delta
    elif greeks == "gamma":
        return -p.gamma
    elif greeks == "theta":
        return -p.theta
    elif greeks == "vega":
        return -p.vega
    else:
        return (strike + issue_price - 1) * np.exp(-r * T_years) - p.price


Sarr = np.linspace(0.8, 1.3, 1000)
t_range = range(0, 65, 16)
params = dict(
    strike=1.0350,
    issue_price=1.0172,
)


def plot_S_t_price_delta():

    eu_bs.plot_sensitivity(
        reln_pricing,
        Sarr,
        t_range,
        params,
        draw_other_lines="\n".join(
            [
                "r = func_other_params['strike'] + func_other_params['issue_price'] - 1",
                "ax.axhline(r, 0, 0.6, ls='--', lw=1)",
                """ax.annotate(f'strike + issue_price - 1 = {r}', xy=(0.78, r+0.005), rotation=0)""",
            ]
        ),
    )
    plt.savefig(EXAMPLT2_IMG / "reln_price.png", bbox_inches="tight", dpi=200)



eu_bs.plot_sigma_sensitivity(
    reln_pricing,
    Sarr,
    np.arange(0.15, 0.3, 0.05),
    params,
)
plt.show()
