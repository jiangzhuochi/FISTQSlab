from dataclasses import replace
from typing import Literal

import numpy as np
import pandas as pd

from .euro_option_bs import *

FIELD = Literal["price", "delta", "gamma", "theta", "vega"]

# TODO: 欧式期权BS公式定价接入flask, 设计api


def vanilla_price_series(S: int | float, L, T, r, sigma, field: FIELD) -> pd.DataFrame:
    """BS公式计算一系列标的现价不同的普通欧式期权价格。

    Parameters
    ----------
    L : int | float
        执行价
    S : int | float
        标的现价
    T : int | float
        有效期(单位: 年), 注: 期权有效天数与365的比值
    r : int | float
        连续复利计无风险利率, 注: 如果年复利利率为r0, 则连续复利利率为ln(1+r0)
    sigma : int | float
        年化标准差
    field : FIELD
        要计算的指标, 价格或希腊字母

    Returns
    -------
    pd.DataFrame
        欧式看涨和看跌期权的指定field
    """

    c = EuropeanCallOptionBS(S=S, L=L, T=T, r=r, sigma=sigma)
    p = EuropeanPutOptionBS(S=S, L=L, T=T, r=r, sigma=sigma)

    low, high = np.floor(S - 10), np.ceil(S + 10)
    S_ls = np.arange(low, high, 0.05)
    C_ls = map(lambda S: getattr(replace(c, S=S), field), S_ls)
    P_ls = map(lambda S: getattr(replace(p, S=S), field), S_ls)
    df = pd.DataFrame({"call": C_ls, "put": P_ls}, index=S_ls)

    return df
