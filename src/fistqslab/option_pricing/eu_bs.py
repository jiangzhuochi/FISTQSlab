from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich
from scipy.stats import norm


@dataclass
class EuropeanOption:
    """欧式期权, BS公式定价"""

    S: np.ndarray  # 标的现价
    L: float  # 执行价
    T_years: float  # 有效期(按年计)
    b: float  # 持有成本率,  无股息时b=r, 连续股利时b=r-q
    r: float  # 连续复利无风险利率, 若年复利无风险利率为r0, 则r = ln(1+r0)
    sigma: float  # 年化标准差

    def __post_init__(self):

        self.d1 = (
            (np.log(self.S / self.L) + (self.b + 0.5 * self.sigma**2) * self.T_years)
            / (self.sigma * np.sqrt(self.T_years))
            if self.T_years != 0
            # 当t趋向于到期时, 实值期权d1趋向于负无穷, 平值期权d1趋向于0, 虚值d1趋向于正无穷
            else np.piecewise(
                self.S,
                [self.S < self.L, self.S == self.L, self.S > self.L],
                [-np.inf, 0, np.inf],
            )
        )
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T_years)
        self.N = norm.cdf
        self.n = norm.pdf
        self.discount = np.exp((self.b - self.r) * self.T_years)

    @property
    def gamma(self):
        return (
            self.discount
            * self.n(self.d1)
            / (self.S * self.sigma * np.sqrt(self.T_years))
            if self.T_years != 0
            # 当t趋向于到期时, 平值期权gamma不存在
            else np.piecewise(
                self.S,
                [self.S == self.L, self.S != self.L],
                [np.nan, 0],
            )
        )

    @property
    def vega(self):
        return self.discount * self.S * np.sqrt(self.T_years) * self.n(self.d1)


@dataclass
class EuropeanCallOption(EuropeanOption):
    """欧式认购期权, BS公式定价"""

    @property
    def price(self):
        return self.S * self.discount * self.N(self.d1) - self.L * np.exp(
            -self.r * self.T_years
        ) * self.N(self.d2)

    @property
    def delta(self):
        return (
            self.discount * self.N(self.d1)
            if self.T_years != 0
            # 当t趋向于到期时, 平值期权delta不存在
            else np.piecewise(
                self.S,
                [self.S < self.L, self.S == self.L, self.S > self.L],
                [0, -np.nan, 1],
            )
        )

    @property
    def theta(self):
        return (
            -self.discount
            * self.S
            * self.sigma
            * self.n(self.d1)
            / (2 * np.sqrt(self.T_years))
            - (self.b - self.r) * self.S * self.discount * self.N(self.d1)
            - self.L * self.r * np.exp(-self.r * self.T_years) * self.N(self.d2)
            if self.T_years != 0
            # 当t趋向于到期时, 实值期权theta趋向于self.L*self.r, 平值期权theta趋向于负无穷, 虚值theta趋向于0
            else np.piecewise(
                self.S,
                [self.S < self.L, self.S == self.L, self.S > self.L],
                [0, -np.inf, -self.L * self.r],
            )
        )

    @property
    def rho(self):
        return (
            self.T_years * self.L * np.exp(-self.r * self.T_years) * self.N(self.d2)
            if self.T_years != 0
            # TODO FIXME 极限情况可能错误，建议不要填入T_years=0，而是很小但不等于0的数
            else np.piecewise(
                self.S,
                [self.S < self.L, self.S == self.L, self.S > self.L],
                [0, -np.nan, 1],
            )
        )


@dataclass
class EuropeanPutOption(EuropeanOption):
    """欧式认沽期权, BS公式定价"""

    @property
    def price(self):
        return self.L * np.exp(-self.r * self.T_years) * self.N(
            -self.d2
        ) - self.S * self.discount * self.N(-self.d1)

    @property
    def delta(self):
        return (
            self.discount * (self.N(self.d1) - 1)
            if self.T_years != 0
            # 当t趋向于到期时, 平值期权delta不存在
            else np.piecewise(
                self.S,
                [self.S < self.L, self.S == self.L, self.S > self.L],
                [-1, -np.nan, 0],
            )
        )

    @property
    def theta(self):
        return (
            -self.discount
            * self.S
            * self.sigma
            * self.n(self.d1)
            / (2 * np.sqrt(self.T_years))
            + (self.b - self.r) * self.S * self.discount * self.N(-self.d1)
            + self.L * self.r * np.exp(-self.r * self.T_years) * self.N(-self.d2)
            if self.T_years != 0
            # 当t趋向于到期时, 实值期权theta趋向于self.L*self.r, 平值期权theta趋向于负无穷, 虚值theta趋向于0
            else np.piecewise(
                self.S,
                [self.S < self.L, self.S == self.L, self.S > self.L],
                [self.L * self.r, -np.inf, 0],
            )
        )

    @property
    def rho(self):
        return (
            -self.T_years * self.L * np.exp(-self.r * self.T_years) * self.N(-self.d2)
            if self.T_years != 0
            # TODO FIXME 极限情况可能错误，建议不要填入T_years=0，而是很小但不等于0的数
            else np.piecewise(
                self.S,
                [self.S < self.L, self.S == self.L, self.S > self.L],
                [0, -np.nan, 1],
            )
        )

@dataclass
class CashOrNothingOption(EuropeanOption):
    """现金或无期权"""

    # 现金数量
    K: float
    # 期权类型, 认购为 1, 认沽为 -1
    option_type: int

    @property
    def price(self):
        return (
            self.K * np.exp(-self.r * self.T_years) * self.N(self.option_type * self.d2)
        )

    @property
    def delta(self):
        return (
            self.K
            * self.option_type
            * np.exp(-self.r * self.T_years)
            * self.n(self.d2)
            / (self.S * self.sigma * np.sqrt(self.T_years))
            if self.T_years != 0
            # 当t趋向于到期时, 平值期权delta不存在
            else np.piecewise(
                self.S,
                [self.S == self.L, self.S != self.L],
                [-np.nan, 0],
            )
        )

    @property
    def gamma(self):
        return (
            -self.K
            * self.option_type
            * np.exp(-self.r * self.T_years)
            * self.n(self.d2)
            * self.d1
            / (self.S**2 * self.sigma**2 * self.T_years)
            if self.T_years != 0
            # 当t趋向于到期时, 平值期权gamma不存在
            else np.piecewise(
                self.S,
                [self.S == self.L, self.S != self.L],
                [np.nan, 0],
            )
        )

    @property
    def vega(self):
        return (
            (
                -self.K
                * self.option_type
                * np.exp(-self.r * self.T_years)
                * self.n(self.d2)
                * self.d1
                / self.sigma
            )
            if self.T_years != 0
            # 当t趋向于到期时, vega趋向于0
            else np.zeros_like(self.S)
        )

    @property
    def theta(self):
        # https://www.studocu.com/en-gb/document/university-of-sheffield/business-economics/fm2014-s-ch16-cash-or-nothing/1522887
        # https://quantpie.co.uk/bsm_bin_c_formula/bs_bin_c_summary.php
        # 里面 rd 是无风险收益率 rf 是股息
        # 期权定价公式指南里的 b = rd - rf, r = rd
        # rd = b + rf
        # https://quantpie.co.uk/bsm_formula/bs_summary.php
        # rf should be interpreted as the dividend yield rate and  rd as the interest/discount rate.
        return (
            self.K
            * np.exp(-self.r * self.T_years)
            * (
                self.option_type
                # * self.n(self.d2 * self.option_type) 是否有 * self.option_type？？？
                * self.n(self.d2)
                * (
                    self.d1 / (2 * self.T_years)
                    - self.b / (self.sigma * np.sqrt(self.T_years))
                )
                + self.r * self.N(self.option_type * self.d2)
            )
            if self.T_years != 0
            else np.piecewise(
                self.S,
                [self.S == self.L, self.S != self.L],
                [np.inf, 0],
            )
        )

    @property
    def rho(self):
        return (
            self.K
            * np.exp(-self.r * self.T_years)
            * (
                self.option_type * np.sqrt(self.T_years) * self.n(self.d2) / self.sigma
                - self.T_years * self.N(self.option_type * self.d2)
            )
            if self.T_years != 0
            # TODO FIXME 极限情况可能错误，建议不要填入T_years=0，而是很小但不等于0的数
            else np.piecewise(
                self.S,
                [self.S == self.L, self.S != self.L],
                [np.inf, 0],
            )
        )

    def show(self):
        res = {
            "NPV": self.price,
            "Delta": self.delta,
            "Gamma": self.gamma,
            "Vega": self.vega,
            "Rho": self.rho,
        }
        for k, v in res.items():
            rich.print(f"[blue]{k}[/blue] = {v}")


def plot_sensitivity(
    func, Sarr, t_range, func_other_params={}, draw_other_lines="pass"
):
    fig, ax = plt.subplots(1, 1, figsize=(12, 7.5))
    exec(draw_other_lines)
    pd.DataFrame(
        [func(Sarr, t=t, **func_other_params) for t in t_range],
        index=[f"t={t}" for t in t_range],
        columns=Sarr,
    ).T.plot(ax=ax)
    ax.set_xlabel("S")
    ax.set_ylabel(func_other_params.get("greeks", "price"))


def plot_sigma_sensitivity(
    func, Sarr, sigma_range, func_other_params={}, draw_other_lines="pass"
):
    fig, ax = plt.subplots(1, 1, figsize=(12, 7.5))
    exec(draw_other_lines)
    pd.DataFrame(
        [func(Sarr, sigma=sigma, **func_other_params) for sigma in sigma_range],
        index=[f"sigma={round(sigma,2)}" for sigma in sigma_range],
        columns=Sarr,
    ).T.plot(ax=ax)
    ax.axvline(1, 0, 1, ls="--", lw=1)
    ax.set_xlabel("S")
    ax.set_ylabel(func_other_params.get("greeks", "price"))
