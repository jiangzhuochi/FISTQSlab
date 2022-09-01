from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .abc_option import BaseOption, Call, Put

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

    S: int | float  # 标的现价
    L: int | float  # 执行价
    T: int | float  # 有效期(单位: 年), 期权有效天数与365的比值
    r: int | float  # 连续复利无风险利率, 若年复利无风险利率为r0, 则r = ln(1+r0)
    sigma: int | float  # 年化标准差

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
