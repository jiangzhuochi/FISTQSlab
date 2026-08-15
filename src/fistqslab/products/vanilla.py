"""欧式香草期权产品（call / put / digital）。"""

from __future__ import annotations

import numpy as np

from fistqslab.market.state import MarketState
from fistqslab.models.black_scholes import (
    bs_call,
    bs_digital_call,
    bs_digital_put,
    bs_put,
)
from fistqslab.products.base import Product

__all__ = ["EuropeanCallOption", "EuropeanPutOption", "DigitalCallOption", "DigitalPutOption"]


def _check_single_asset(n_assets: int) -> None:
    if n_assets != 1:
        raise ValueError(f"香草期权仅支持单标的，当前 {n_assets} 个标的")


class EuropeanCallOption(Product):
    """欧式认购期权（按金额计价，payoff = max(S_T - K, 0)）。"""

    def __init__(self, strike: float, maturity_year_fraction: float):
        self.strike = float(strike)
        self._maturity = float(maturity_year_fraction)

    @property
    def maturity_year_fraction(self) -> float:
        return self._maturity

    def payoff_terminal(self, rel: np.ndarray, spots: np.ndarray) -> np.ndarray:
        _check_single_asset(rel.shape[0])
        s_t = rel[0] * spots[0]
        return np.maximum(s_t - self.strike, 0.0)

    def analytic_price(self, market: MarketState) -> float | None:
        _check_single_asset(market.n_assets)
        return bs_call(
            market.spots[0],
            self.strike,
            self._maturity,
            market.risk_free_rate,
            market.volatilities[0],
            market.dividend_yields[0],
        )


class EuropeanPutOption(Product):
    """欧式认沽期权（按金额计价，payoff = max(K - S_T, 0)）。"""

    def __init__(self, strike: float, maturity_year_fraction: float):
        self.strike = float(strike)
        self._maturity = float(maturity_year_fraction)

    @property
    def maturity_year_fraction(self) -> float:
        return self._maturity

    def payoff_terminal(self, rel: np.ndarray, spots: np.ndarray) -> np.ndarray:
        _check_single_asset(rel.shape[0])
        s_t = rel[0] * spots[0]
        return np.maximum(self.strike - s_t, 0.0)

    def analytic_price(self, market: MarketState) -> float | None:
        _check_single_asset(market.n_assets)
        return bs_put(
            market.spots[0],
            self.strike,
            self._maturity,
            market.risk_free_rate,
            market.volatilities[0],
            market.dividend_yields[0],
        )


class DigitalCallOption(Product):
    """现金或无认购期权（payoff = cash · 1{S_T > K}）。"""

    def __init__(
        self,
        strike: float,
        maturity_year_fraction: float,
        cash: float = 1.0,
    ):
        self.strike = float(strike)
        self.cash = float(cash)
        self._maturity = float(maturity_year_fraction)

    @property
    def maturity_year_fraction(self) -> float:
        return self._maturity

    def payoff_terminal(self, rel: np.ndarray, spots: np.ndarray) -> np.ndarray:
        _check_single_asset(rel.shape[0])
        s_t = rel[0] * spots[0]
        return np.where(s_t > self.strike, self.cash, 0.0)

    def analytic_price(self, market: MarketState) -> float | None:
        _check_single_asset(market.n_assets)
        return bs_digital_call(
            market.spots[0],
            self.strike,
            self._maturity,
            market.risk_free_rate,
            market.volatilities[0],
            market.dividend_yields[0],
            cash=self.cash,
        )


class DigitalPutOption(Product):
    """现金或无认沽期权（payoff = cash · 1{S_T < K}）。"""

    def __init__(
        self,
        strike: float,
        maturity_year_fraction: float,
        cash: float = 1.0,
    ):
        self.strike = float(strike)
        self.cash = float(cash)
        self._maturity = float(maturity_year_fraction)

    @property
    def maturity_year_fraction(self) -> float:
        return self._maturity

    def payoff_terminal(self, rel: np.ndarray, spots: np.ndarray) -> np.ndarray:
        _check_single_asset(rel.shape[0])
        s_t = rel[0] * spots[0]
        return np.where(s_t < self.strike, self.cash, 0.0)

    def analytic_price(self, market: MarketState) -> float | None:
        _check_single_asset(market.n_assets)
        return bs_digital_put(
            market.spots[0],
            self.strike,
            self._maturity,
            market.risk_free_rate,
            market.volatilities[0],
            market.dividend_yields[0],
            cash=self.cash,
        )
