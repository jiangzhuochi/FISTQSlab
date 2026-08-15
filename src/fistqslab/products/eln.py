"""ELN / RELN（反向可转债类）产品。

以 `example2/` 的闭式分解为权威定义（作者本意语义）：

- **ELN**：以 ``issue_price`` 买入 1 份 ELN
  = 持有到期收到 1 的票据 + ``1/strike`` 份欧式认沽期权空头

  .. math::
      payoff = 1 - \\frac{(strike - \\min_i S_T^i/S_0^i)^+}{strike}

- **RELN**：以 1 买入 1 份 RELN
  = 持有到期支付 ``strike + issue_price - 1`` 的票据 + 1 份认沽空头

  .. math::
      payoff = (strike + issue_price - 1) - (strike - \\min_i S_T^i/S_0^i)^+

所有参数为相对比例（相对期初 spot）。票息为整个期限的总票息。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve

from fistqslab.market.state import MarketState
from fistqslab.models.black_scholes import bs_put
from fistqslab.products.base import Product

__all__ = [
    "ELN",
    "RELN",
    "get_eln_strike_from_issue_price",
    "get_reln_issue_price",
]


def _worst(rel: np.ndarray) -> np.ndarray:
    """worst-of 相对收益（多标的取最小）。"""
    return np.min(rel, axis=0)


class ELN(Product):
    """反向可转债（Equity-Linked Note）。

    Args:
        strike: 行权价（相对期初 spot 的比例）。
        issue_price: 发行价（相对 1 元本金）；不参与 payoff，仅记录。
        maturity_year_fraction: 到期年数。
    """

    def __init__(
        self,
        strike: float,
        maturity_year_fraction: float,
        issue_price: float = 1.0,
    ):
        if strike <= 0:
            raise ValueError("strike 必须为正")
        self.strike = float(strike)
        self.issue_price = float(issue_price)
        self._maturity = float(maturity_year_fraction)

    @property
    def maturity_year_fraction(self) -> float:
        return self._maturity

    def payoff_terminal(self, rel: np.ndarray, spots: np.ndarray) -> np.ndarray:
        w = _worst(rel)
        return 1.0 - np.maximum(self.strike - w, 0.0) / self.strike

    def analytic_price(self, market: MarketState) -> float | None:
        if market.n_assets > 1:
            # worst-of 多标的需要多元正态积分，暂不提供闭式
            return None
        T = self._maturity
        r = market.risk_free_rate
        p = bs_put(
            1.0,
            self.strike,
            T,
            r,
            market.volatility_vector[0],
            market.dividend_vector[0],
        )
        return np.exp(-r * T) - p / self.strike


class RELN(Product):
    """逆向反向可转债（Reverse ELN）。

    Args:
        strike: 行权价（相对比例）。
        issue_price: 发行价（相对 1 元本金）。
        maturity_year_fraction: 到期年数。
    """

    def __init__(
        self,
        strike: float,
        issue_price: float,
        maturity_year_fraction: float,
    ):
        if strike <= 0:
            raise ValueError("strike 必须为正")
        self.strike = float(strike)
        self.issue_price = float(issue_price)
        self._maturity = float(maturity_year_fraction)

    @property
    def maturity_year_fraction(self) -> float:
        return self._maturity

    def payoff_terminal(self, rel: np.ndarray, spots: np.ndarray) -> np.ndarray:
        w = _worst(rel)
        return (self.strike + self.issue_price - 1.0) - np.maximum(self.strike - w, 0.0)

    def analytic_price(self, market: MarketState) -> float | None:
        if market.n_assets > 1:
            return None
        T = self._maturity
        r = market.risk_free_rate
        p = bs_put(
            1.0,
            self.strike,
            T,
            r,
            market.volatility_vector[0],
            market.dividend_vector[0],
        )
        return (self.strike + self.issue_price - 1.0) * np.exp(-r * T) - p


def get_eln_strike_from_issue_price(
    issue_price: float,
    maturity_year_fraction: float,
    market: MarketState,
    x0: float = 0.95,
) -> float:
    """反解使 ELN 理论价格等于给定发行价的 strike。

    Args:
        issue_price: 目标发行价（相对 1 元本金）。
        maturity_year_fraction: 到期年数。
        market: 市场参数（仅支持单标的）。
        x0: fsolve 初值。

    Returns:
        满足 ``ELN(strike).analytic_price(market) == issue_price`` 的 strike。
    """
    if market.n_assets > 1:
        raise ValueError("反解仅支持单标的")

    def f(strike: float) -> float:
        # fsolve 回调可能传入 shape (1,) 的数组（scipy >= 1.17），需先收敛为标量
        strike_f = float(np.asarray(strike).item())
        eln = ELN(strike_f, maturity_year_fraction)
        return float(eln.analytic_price(market) - issue_price)  # type: ignore[union-attr]

    sol = np.asarray(fsolve(f, x0=x0, xtol=1e-8))
    return float(np.atleast_1d(sol)[0])


def get_reln_issue_price(
    strike: float,
    maturity_year_fraction: float,
    market: MarketState,
) -> float:
    """计算使 RELN 平价发行（价格 = 1）的 issue_price。

    Args:
        strike: 行权价。
        maturity_year_fraction: 到期年数。
        market: 市场参数（仅支持单标的）。

    Returns:
        平价发行价。
    """
    if market.n_assets > 1:
        raise ValueError("反解仅支持单标的")
    T = maturity_year_fraction
    r = market.risk_free_rate
    p = bs_put(
        1.0, strike, T, r, market.volatility_vector[0], market.dividend_vector[0]
    )
    # price(strike, issue) = (strike + issue - 1)·e^{-rT} - p = 1
    return (1.0 + p) / np.exp(-r * T) - strike + 1.0
