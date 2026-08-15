"""Rainbow Note（worst-of 增利票据）产品。

以 `example2/rb.py` 的闭式分解为权威定义：

.. math::
    payoff = 1 + coupon + participation \\cdot (w - lcs)^+ - \\frac{(ps - w)^+}{ps}

其中 :math:`w = \\min_i S_T^i/S_0^i`，:math:`ps` = put_strike，
:math:`lcs` = lower_call_strike（须 :math:`\\ge ps`），
:math:`participation` = upside_participation。

**重构说明**：旧实现中 ``lower_call_strike`` 与 ``upside_participation``
未参与定价（``price()`` 只按 put_strike 与 coupon 计算），本模块将其
真正纳入 payoff；二者不再能忽略。
"""

from __future__ import annotations

import numpy as np

from fistqslab.market.state import MarketState
from fistqslab.models.black_scholes import bs_call, bs_put
from fistqslab.products.base import Product

__all__ = ["RainbowNote"]


class RainbowNote(Product):
    """worst-of Rainbow Note。

    Args:
        put_strike: 转股行权价（相对比例）。
        lower_call_strike: 上行认购行权价（相对比例，须 >= put_strike）。
        upside_participation: 上行参与率。
        guaranteed_flat_coupon: 保证票息（整个期限总票息）。
        maturity_year_fraction: 到期年数。
    """

    def __init__(
        self,
        put_strike: float,
        lower_call_strike: float,
        upside_participation: float,
        guaranteed_flat_coupon: float,
        maturity_year_fraction: float,
    ):
        if put_strike <= 0:
            raise ValueError("put_strike 必须为正")
        if lower_call_strike < put_strike:
            raise ValueError("lower_call_strike 必须 >= put_strike")
        if upside_participation < 0:
            raise ValueError("upside_participation 必须非负")
        self.put_strike = float(put_strike)
        self.lower_call_strike = float(lower_call_strike)
        self.upside_participation = float(upside_participation)
        self.guaranteed_flat_coupon = float(guaranteed_flat_coupon)
        self._maturity = float(maturity_year_fraction)

    @property
    def maturity_year_fraction(self) -> float:
        return self._maturity

    def payoff_terminal(self, rel: np.ndarray, spots: np.ndarray) -> np.ndarray:
        w = np.min(rel, axis=0)
        ps = self.put_strike
        lcs = self.lower_call_strike
        part = self.upside_participation
        c = self.guaranteed_flat_coupon
        return (
            1.0
            + c
            + part * np.maximum(w - lcs, 0.0)
            - np.maximum(ps - w, 0.0) / ps
        )

    def analytic_price(self, market: MarketState) -> float | None:
        if market.n_assets > 1:
            return None
        T = self._maturity
        r = market.risk_free_rate
        sigma = market.volatilities[0]
        q = market.dividend_yields[0]
        c = bs_call(1.0, self.lower_call_strike, T, r, sigma, q)
        p = bs_put(1.0, self.put_strike, T, r, sigma, q)
        return (
            np.exp(-r * T) * (1.0 + self.guaranteed_flat_coupon)
            + self.upside_participation * c
            - p / self.put_strike
        )
