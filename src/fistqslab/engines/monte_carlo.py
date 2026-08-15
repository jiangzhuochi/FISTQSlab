"""分块 Monte Carlo 定价引擎。

与旧实现的本质区别：

- **内存恒定**：按 ``batch_size`` 分块消费模拟器产出，只保留
  ``sum(payoff)`` / ``sum(payoff²)`` / ``n`` 三个标量累加器；
  100 万条路径的内存占用与 1 万条相同（数十 MB 量级）。
- **结果完整**：返回 ``PricingResult``，含价格、标准误与 95% 置信区间
  （MC 结果没有 stderr 是不完整的）。
- **可复现**：相同 ``seed`` 产出逐位一致的结果，也是
  Common Random Numbers（CRN）Greeks 的基础。
"""

from __future__ import annotations

import numpy as np

from fistqslab.models.gbm import GBMModel
from fistqslab.products.base import Product
from fistqslab.results import PricingResult


class _Accumulator:
    """只保留 sum / sum² / n 的方差累加器。"""

    __slots__ = ("sum", "sum2", "n")

    def __init__(self) -> None:
        self.sum = 0.0
        self.sum2 = 0.0
        self.n = 0

    def update(self, payoffs: np.ndarray) -> None:
        self.sum += float(payoffs.sum())
        self.sum2 += float(np.square(payoffs).sum())
        self.n += int(payoffs.size)

    @property
    def mean(self) -> float:
        if self.n == 0:
            raise RuntimeError("累加器为空")
        return self.sum / self.n

    @property
    def stderr(self) -> float:
        """样本均值标准误 sqrt(var / n)。"""
        if self.n < 2:
            return float("nan")
        var = self.sum2 / self.n - self.mean**2
        return float(np.sqrt(max(var, 0.0) / self.n))


class MonteCarloEngine:
    """分块 Monte Carlo 定价引擎。

    Args:
        n_paths: 独立随机抽样数（对偶变量时有效路径数为 2×）。
        batch_size: 每批独立抽样数，控制内存上界。
        antithetic: 是否使用对偶变量方差缩减。
        seed: 随机种子；相同 seed 结果逐位一致。
    """

    def __init__(
        self,
        n_paths: int = 100_000,
        batch_size: int = 20_000,
        antithetic: bool = True,
        seed: int | None = None,
    ):
        if n_paths <= 0:
            raise ValueError("n_paths 必须为正")
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正")
        self.n_paths = n_paths
        self.batch_size = batch_size
        self.antithetic = antithetic
        self.seed = seed

    def price(self, product: Product, model: GBMModel) -> PricingResult:
        """Monte Carlo 定价。

        产品仅依赖终值时走 ``payoff_terminal``（未折现，引擎统一折现）；
        有观察日时走 ``payoff_paths``（产品自行按观察日折现）。

        Args:
            product: 产品。
            model: GBM 模型（含市场参数）。

        Returns:
            ``PricingResult``，含 stderr 与 95% CI。
        """
        market = model.market
        obs = product.observation_year_fractions
        acc = _Accumulator()

        if obs is None:
            # terminal-only：payoff 未折现，最后统一折现
            for rel, _ in model.terminal_batches(
                product.maturity_year_fraction,
                self.n_paths,
                self.batch_size,
                self.antithetic,
                self.seed,
            ):
                acc.update(product.payoff_terminal(rel, market.spot_vector))
            discount = np.exp(-market.risk_free_rate * product.maturity_year_fraction)
            price = discount * acc.mean
        else:
            for rel, _ in model.path_batches(
                obs,
                self.n_paths,
                self.batch_size,
                self.antithetic,
                self.seed,
            ):
                acc.update(product.payoff_paths(rel, market.spot_vector, market))
            price = acc.mean

        return PricingResult(
            price=float(price),
            stderr=acc.stderr,
            n_paths=self.n_paths,
            n_effective_paths=acc.n,
            seed=self.seed,
            method="monte_carlo",
        )
