from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import norm

from .abc_option import FIELD, BaseOption, Call, Put

# TODO: 欧式期权BS公式定价接入flask, 设计api


"""notes:
为了简便, 暂时将欧式期权的定价与连续BS公式定价耦合。

在这里每个类中的方法就相当于连续BS公式定价
如果要离散定价, 那可以另写一个文件
在这里定价模型是主要的, 而保存期权数据是次要的
于是感觉将数据和方法解耦意义不大
"""


@dataclass
class EuropeanOptionBS(BaseOption):
    """抽象类, 欧式期权, BS公式定价"""

    S: float  # 标的现价
    L: float  # 执行价
    T: float  # 有效期(单位: 年), 期权有效天数与365的比值
    r: float  # 连续复利无风险利率, 若年复利无风险利率为r0, 则r = ln(1+r0)
    sigma: float  # 年化标准差

    def __post_init__(self):
        self.d1 = (
            np.log(self.S / self.L) + (self.r + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)
        self.N = norm.cdf
        self.n = norm.pdf


@dataclass
class EuropeanCallOptionBS(EuropeanOptionBS, Call):
    """欧式看涨期权, BS公式定价"""

    @property
    def price(self):
        return self.S * self.N(self.d1) - self.L * np.exp(-self.r * self.T) * self.N(
            self.d2
        )

    @property
    def delta(self):
        return self.N(self.d1)

    @property
    def gamma(self):
        return self.n(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    @property
    def theta(self):
        return -(
            self.S * self.sigma * self.n(self.d1) / (2 * np.sqrt(self.T))
            + self.L * self.r * np.exp(-self.r * self.T) * self.N(self.d2)
        )

    @property
    def vega(self):
        return self.S * np.sqrt(self.T) * self.n(self.d1)


@dataclass
class EuropeanPutOptionBS(EuropeanOptionBS, Put):
    """欧式看跌期权, BS公式定价"""

    @property
    def price(self):
        return self.L * np.exp(-self.r * self.T) * self.N(-self.d2) - self.S * self.N(
            -self.d1
        )

    @property
    def delta(self):
        return self.N(self.d1) - 1

    @property
    def gamma(self):
        return self.n(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    @property
    def theta(self):
        return self.L * self.r * np.exp(-self.r * self.T) * self.N(
            -self.d2
        ) - self.S * self.sigma * self.n(self.d1) / (2 * np.sqrt(self.T))

    @property
    def vega(self):
        return self.S * np.sqrt(self.T) * self.n(self.d1)


# 「开始计算」时使用, 返回单只期权(包括看涨和看跌)价格和所有希腊字母
def euro_option_bs():
    pass


# 绘图时使用, 返回标的价格不同的一系列期权(包括看涨和看跌)的价格或某个希腊字母
def euro_option_bs_series(S: float, L, T, r, sigma, field: FIELD) -> pd.DataFrame:
    """BS公式计算一系列标的现价不同的普通欧式期权价格。

    Parameters
    ----------
    L : float
        执行价
    S : float
        标的现价
    T : float
        有效期(单位: 年), 注: 期权有效天数与365的比值
    r : float
        连续复利计无风险利率, 注: 如果年复利利率为r0, 则连续复利利率为ln(1+r0)
    sigma : float
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
    df = (
        pd.DataFrame({"call": C_ls, "put": P_ls}, index=S_ls)
        .rename(index=lambda n: f"{n:.2f}")
        .applymap(lambda n: np.around(n, 4))
    )

    return df
