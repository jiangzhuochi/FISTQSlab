from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def eln_pricing(
    S=np.array([0.8, 0.94, 1]),
    strike=0.9404,
    issue_price=0.9828,
    t=0,
    T_days=64,
    sigma=0.228677,
    r=np.log(1.015),
    greeks=None,
):
    """ELN 定价

    以issue_price买入1份ELN = 持有1份到期收到1的票据 + 1/strike份欧式认沽期权空头"""

    T_years = (T_days - t) / 365
    p = eu_bs.EuropeanPutOption(S=S, L=strike, T_years=T_years, r=r, sigma=sigma)
    if greeks == "delta":
        return -(1 / strike) * p.delta
    elif greeks == "gamma":
        return -(1 / strike) * p.gamma
    elif greeks == "theta":
        return -(1 / strike) * p.theta
    elif greeks == "vega":
        return -(1 / strike) * p.vega
    else:
        return 1 * np.exp(-r * T_years) - (1 / strike) * p.price


Sarr = np.linspace(0.7, 1.2, 1000)
t_range = range(0, 65, 16)
params = dict(strike=0.9404, issue_price=0.9828)
eu_bs.plot_sensitivity(
    eln_pricing,
    Sarr,
    t_range,
    params,
)
plt.savefig(EXAMPLT2_IMG / "eln_price.png", bbox_inches="tight", dpi=200)
plt.show()


eu_bs.plot_sensitivity(
    eln_pricing,
    Sarr,
    t_range,
    {**params, "greeks": "delta"},
    draw_other_lines="\n".join(
        [
            "strike = func_other_params['strike']",
            "ax.axhline(1/strike, 0, 0.3, ls='--', lw=1)",
            """ax.annotate(f'1 / strike = {round(1/strike, 3)}', xy=(0.68,  1 / strike+0.02), rotation=0)""",
            "ax.axvline(strike, 0, 0.3, ls='--', lw=1)",
            """ax.annotate(f'strike = {strike}', xy=(strike-0.02, 0), rotation=270)""",
        ]
    ),
)
plt.savefig(EXAMPLT2_IMG / "eln_delta.png", bbox_inches="tight", dpi=200)
plt.show()
