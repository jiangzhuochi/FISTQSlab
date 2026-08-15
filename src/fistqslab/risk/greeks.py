"""风险层：有限差分希腊字母（Common Random Numbers）。

对每个标的做 ±ε 的中心差分并**用相同 seed 重新定价**，保证上下两组
路径逐位同源（Common Random Numbers），差分的蒙特卡洛噪声被显著压低。

注意：bump 的是期初 spot（``market.spots``），利率/波动率/相关性不变。
对 terminal-only 产品，折现因子不变，价格差异仅来自 payoff。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fistqslab.engines.monte_carlo import MonteCarloEngine
from fistqslab.models.gbm import GBMModel
from fistqslab.products.base import Product


class GreeksCalculator:
    """有限差分 Greeks 计算器。

    Args:
        engine: Monte Carlo 引擎（其 ``seed`` 用于 CRN）。
        epsilon: 差分步长（相对 spot 的比例）。
    """

    def __init__(self, engine: MonteCarloEngine, epsilon: float = 1e-3):
        if epsilon <= 0:
            raise ValueError("epsilon 必须为正")
        self.engine = engine
        self.epsilon = float(epsilon)

    def _price_with_bump(self, product: Product, model: GBMModel, i: int, eps: float) -> float:
        market = model.market
        bumped_spots = market.spots.copy()
        bumped_spots[i] *= 1.0 + eps
        bumped = replace(market, spots=bumped_spots)
        return float(self.engine.price(product, GBMModel(bumped)).price)

    def delta(self, product: Product, model: GBMModel) -> np.ndarray:
        """一阶希腊字母：dV/dS（对每标的绝对价格）。"""
        n = model.n_assets
        eps = self.epsilon
        out = np.empty(n)
        for i in range(n):
            pu = self._price_with_bump(product, model, i, eps)
            pd = self._price_with_bump(product, model, i, -eps)
            out[i] = (pu - pd) / (2 * eps * model.market.spots[i])
        return out

    def gamma(self, product: Product, model: GBMModel) -> np.ndarray:
        """二阶希腊字母：d²V/dS²（对每标的绝对价格）。"""
        n = model.n_assets
        eps = self.epsilon
        base = float(self.engine.price(product, model).price)
        out = np.empty(n)
        for i in range(n):
            pu = self._price_with_bump(product, model, i, eps)
            pd = self._price_with_bump(product, model, i, -eps)
            out[i] = (pu - 2 * base + pd) / (eps**2 * model.market.spots[i] ** 2)
        return out

    def delta_gamma(self, product: Product, model: GBMModel) -> tuple[np.ndarray, np.ndarray]:
        """同时计算 delta 与 gamma（共享基础价格，减少一次定价）。"""
        n = model.n_assets
        eps = self.epsilon
        base = float(self.engine.price(product, model).price)
        deltas = np.empty(n)
        gammas = np.empty(n)
        for i in range(n):
            pu = self._price_with_bump(product, model, i, eps)
            pd = self._price_with_bump(product, model, i, -eps)
            deltas[i] = (pu - pd) / (2 * eps * model.market.spots[i])
            gammas[i] = (pu - 2 * base + pd) / (eps**2 * model.market.spots[i] ** 2)
        return deltas, gammas
