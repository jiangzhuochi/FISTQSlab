"""Leverage Note（杠杆票据）产品。

payoff 线性于标的相对收益：

.. math::
    payoff = 1 + leverage \\cdot (S_T/S_0 - 1 + dividend \\cdot T)

其中 ``dividend`` 为年化股息收益率（整个期限按 ``dividend * T`` 计）。

**重构说明**：旧实现的 delta 解析式 ``discount * leverage_multiple``
与其 payoff 定义不一致（payoff 只依赖相对收益，价格对期初 spot 的
导数为 0）。本模块提供一致的解析价格；希腊字母由 ``risk.greeks``
统一以有限差分计算，并与解析行为保持一致。
"""

from __future__ import annotations

import numpy as np

from fistqslab.market.state import MarketState
from fistqslab.products.base import Product

__all__ = ["LeverageNote"]


class LeverageNote(Product):
    """杠杆票据。

    Args:
        leverage_multiple: 杠杆倍数。
        maturity_year_fraction: 到期年数。
        dividend_rate: 年化股息收益率（默认 0）。
    """

    def __init__(
        self,
        leverage_multiple: float,
        maturity_year_fraction: float,
        dividend_rate: float = 0.0,
    ):
        if leverage_multiple <= 0:
            raise ValueError("leverage_multiple 必须为正")
        self.leverage_multiple = float(leverage_multiple)
        self.dividend_rate = float(dividend_rate)
        self._maturity = float(maturity_year_fraction)

    @property
    def maturity_year_fraction(self) -> float:
        return self._maturity

    def payoff_terminal(self, rel: np.ndarray, spots: np.ndarray) -> np.ndarray:
        if rel.shape[0] != 1:
            raise ValueError("LeverageNote 仅支持单标的")
        return 1.0 + self.leverage_multiple * (
            rel[0] - 1.0 + self.dividend_rate * self._maturity
        )

    def analytic_price(self, market: MarketState) -> float | None:
        if market.n_assets != 1:
            return None
        r = market.risk_free_rate
        q = market.dividend_vector[0]
        T = self._maturity
        expected_rel = np.exp((r - q) * T)
        return np.exp(-r * T) * (
            1.0 + self.leverage_multiple * (expected_rel - 1.0 + self.dividend_rate * T)
        )
