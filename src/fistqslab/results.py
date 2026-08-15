"""统一定价结果数据结构。

对 Monte Carlo 结果而言，一个没有 standard error 的价格是不完整的结果。
``PricingResult`` 承载价格、误差与复现信息；解析解结果的 ``stderr`` 为
``None``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 标准正态分布 97.5% 分位点，用于 95% 置信区间
_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class PricingResult:
    """定价结果。

    Attributes:
        price: 产品单位名义本金的现值。
        stderr: Monte Carlo 均值标准误；解析解为 ``None``。
        n_paths: 独立随机样本数（不含 antithetic 翻倍）。
        n_effective_paths: 实际用于估值的路径数（antithetic 时为 2 * n_paths）。
        seed: 随机数种子（用于复现）；解析解为 ``None``。
        method: 定价方法（``"monte_carlo"`` / ``"analytic"``）。
        delta/gamma/vega: 可选希腊字母（对每标的）。
    """

    price: float
    stderr: float | None = None
    n_paths: int | None = None
    n_effective_paths: int | None = None
    seed: int | None = None
    method: str = "monte_carlo"
    delta: np.ndarray | None = None
    gamma: np.ndarray | None = None
    vega: np.ndarray | None = None

    @property
    def ci_low(self) -> float | None:
        """95% 置信区间下界。"""
        if self.stderr is None:
            return None
        return self.price - _Z_95 * self.stderr

    @property
    def ci_high(self) -> float | None:
        """95% 置信区间上界。"""
        if self.stderr is None:
            return None
        return self.price + _Z_95 * self.stderr

    def __str__(self) -> str:  # pragma: no cover - 仅展示辅助
        base = f"PricingResult(price={self.price:.6f}"
        if self.stderr is not None:
            base += (
                f", stderr={self.stderr:.6f}, "
                f"95%CI=[{self.ci_low:.6f}, {self.ci_high:.6f}]"
            )
        if self.n_effective_paths is not None:
            base += f", paths={self.n_effective_paths}"
        if self.seed is not None:
            base += f", seed={self.seed}"
        return base + f", method={self.method!r})"
