"""BEN（增利票据）与 AutoCall（自动赎回票据）产品。

以 `example2/ben.py` 的闭式分解为权威定义：

**无保本 BEN**（``min_redemption=None``）：

.. math::
    payoff = 1 - \\frac{(ps - w)^+}{ps} + (w - (1+c))^+ + c \\cdot \\mathbb{1}\\{w \\ge cb\\}

**保本 BEN**（``min_redemption=mr``）：

.. math::
    payoff = mr + \\frac{(w - ps \\cdot mr)^+ - (w - ps)^+}{ps}
             + (w - (1+c))^+ + c \\cdot \\mathbb{1}\\{w \\ge cb\\}

其中 :math:`w = \\min_i S_T^i/S_0^i`，:math:`ps` = put_strike，
:math:`cb` = coupon_barrier，:math:`c` = bonus_coupon。

**AutoCall**：按观察频率布点；首个观察日 ``w \\ge autocall_barrier``
即提前赎回并按 ``1 + coupon · (t_k/T)`` 支付（按时间 prorata 折现）；
若从未敲出，到期 ``w \\ge ki_barrier`` 支付 ``1 + coupon``，
否则转股 ``max(min_redemption, w / ki_barrier)``。

所有参数为相对比例；票息为整个期限的总票息。
"""

from __future__ import annotations

import numpy as np

from fistqslab.market.day_count import year_fraction
from fistqslab.market.state import MarketState
from fistqslab.models.black_scholes import bs_call, bs_digital_call, bs_put
from fistqslab.products.base import Product

__all__ = ["BonusEnhancedNote", "AutoCallNote"]


def _worst(rel: np.ndarray) -> np.ndarray:
    return np.min(rel, axis=0)


class BonusEnhancedNote(Product):
    """增利票据（Bonus Enhanced Note）。

    Args:
        put_strike: 转股行权价（相对比例）。
        coupon_barrier: 票息触发价（相对比例）。
        bonus_coupon: 票息（整个期限总票息）。
        maturity_year_fraction: 到期年数。
        min_redemption: 最低赎回比例；``None`` 表示无保本。
    """

    def __init__(
        self,
        put_strike: float,
        coupon_barrier: float,
        bonus_coupon: float,
        maturity_year_fraction: float,
        min_redemption: float | None = None,
    ):
        if put_strike <= 0:
            raise ValueError("put_strike 必须为正")
        if min_redemption is not None and not 0 < min_redemption <= 1:
            raise ValueError("min_redemption 须在 (0, 1] 内")
        self.put_strike = float(put_strike)
        self.coupon_barrier = float(coupon_barrier)
        self.bonus_coupon = float(bonus_coupon)
        self.min_redemption = None if min_redemption is None else float(min_redemption)
        self._maturity = float(maturity_year_fraction)

    @property
    def maturity_year_fraction(self) -> float:
        return self._maturity

    def payoff_terminal(self, rel: np.ndarray, spots: np.ndarray) -> np.ndarray:
        w = _worst(rel)
        ps = self.put_strike
        c = self.bonus_coupon
        cb = self.coupon_barrier
        if self.min_redemption is None:
            base = 1.0 - np.maximum(ps - w, 0.0) / ps
        else:
            mr = self.min_redemption
            bull_spread = (np.maximum(w - ps * mr, 0.0) - np.maximum(w - ps, 0.0)) / ps
            base = mr + bull_spread
        bonus = np.maximum(w - (1.0 + c), 0.0)
        coupon = np.where(w >= cb, c, 0.0)
        return base + bonus + coupon

    def analytic_price(self, market: MarketState) -> float | None:
        if market.n_assets > 1:
            return None
        T = self._maturity
        r = market.risk_free_rate
        sigma = market.volatility_vector[0]
        q = market.dividend_vector[0]
        c = self.bonus_coupon

        # 认购多头（行权价 1 + c）
        c3 = bs_call(1.0, 1.0 + c, T, r, sigma, q)
        # 现金或无认购（行权价 coupon_barrier，现金 c）
        con = bs_digital_call(1.0, self.coupon_barrier, T, r, sigma, q, cash=c)
        if self.min_redemption is None:
            p = bs_put(1.0, self.put_strike, T, r, sigma, q)
            return np.exp(-r * T) - p / self.put_strike + c3 + con
        # 保本：牛市价差 (C(ps·mr) - C(ps)) / ps
        c1 = bs_call(1.0, self.put_strike * self.min_redemption, T, r, sigma, q)
        c2 = bs_call(1.0, self.put_strike, T, r, sigma, q)
        return (
            (c1 - c2) / self.put_strike
            + self.min_redemption * np.exp(-r * T)
            + c3
            + con
        )


class AutoCallNote(Product):
    """自动赎回票据（AutoCall）。

    Args:
        ki_barrier: 敲入价（到期失效条件，相对比例）。
        autocall_barrier: 敲出价（观察日提前赎回条件，相对比例）。
        observation_frequency_days: 观察日间隔（自然日）。
        bonus_coupon: 票息（整个期限总票息）。
        maturity_year_fraction: 到期年数。
        min_redemption: 敲入后的最低赎回比例（默认 0 = 无保本）。
    """

    def __init__(
        self,
        ki_barrier: float,
        autocall_barrier: float,
        observation_frequency_days: int,
        bonus_coupon: float,
        maturity_year_fraction: float,
        min_redemption: float = 0.0,
    ):
        if not 0 < ki_barrier <= 1:
            raise ValueError("ki_barrier 须在 (0, 1] 内")
        if not 0 < autocall_barrier <= 1:
            raise ValueError("autocall_barrier 须在 (0, 1] 内")
        if observation_frequency_days <= 0:
            raise ValueError("observation_frequency_days 必须为正")
        if not 0 <= min_redemption <= 1:
            raise ValueError("min_redemption 须在 [0, 1] 内")
        self.ki_barrier = float(ki_barrier)
        self.autocall_barrier = float(autocall_barrier)
        self.observation_frequency_days = int(observation_frequency_days)
        self.bonus_coupon = float(bonus_coupon)
        self.min_redemption = float(min_redemption)
        self._maturity = float(maturity_year_fraction)

        # 观察日（year fraction，含到期日，升序）
        freq = year_fraction(self.observation_frequency_days)
        n_intermediate = int(np.floor(self._maturity / freq))
        obs = np.arange(1, n_intermediate + 1) * freq
        if not np.isclose(obs[-1], self._maturity):
            obs = np.append(obs, self._maturity)
        self._obs = np.asarray(obs, dtype=float)

    @property
    def maturity_year_fraction(self) -> float:
        return self._maturity

    @property
    def observation_year_fractions(self) -> np.ndarray | None:
        return self._obs

    def payoff_terminal(self, rel: np.ndarray, spots: np.ndarray) -> np.ndarray:
        """仅到期观察的退化版本（不提前赎回）。

        实际定价走 ``payoff_paths``；此方法仅为满足接口语义，等价于
        把观察日全部挪到到期日。
        """
        w = np.min(rel, axis=0)
        return np.where(
            w < self.ki_barrier,
            np.maximum(self.min_redemption, w / self.ki_barrier),
            1.0 + self.bonus_coupon,
        )

    def payoff_paths(
        self, rel: np.ndarray, spots: np.ndarray, market: MarketState
    ) -> np.ndarray:
        """AutoCall 收益现值（已按观察日折现）。

        rel: (n_assets, n_obs, n_paths)。更早的观察日优先：一旦某观察日
        worst >= autocall_barrier 即提前赎回，按时间比例 prorata 票息。
        """
        worst = np.min(rel, axis=0)  # (n_obs, n_paths)
        r = market.risk_free_rate
        obs = self._obs
        T = self._maturity

        # 到期 payoff：未敲出则付本金+票息；敲入则转股
        final = np.where(
            worst[-1] < self.ki_barrier,
            np.maximum(self.min_redemption, worst[-1] / self.ki_barrier),
            1.0 + self.bonus_coupon,
        )
        payoff = final * np.exp(-r * T)

        # 从倒数第二个观察日向期初回溯：更早的观察日敲出优先
        for k in range(len(obs) - 2, -1, -1):
            autocalled = worst[k] >= self.autocall_barrier
            value_k = (1.0 + self.bonus_coupon * (obs[k] / T)) * np.exp(-r * obs[k])
            payoff = np.where(autocalled, value_k, payoff)
        return payoff
