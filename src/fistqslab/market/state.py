"""风险中性市场参数容器。

``MarketState`` 承载所有标的与市场层面的输入：价格、无风险利率、股息率、
波动率与相关性。定价函数不再接收散落的 ``r``/``sigma`` 参数，而是接收
一个 ``MarketState``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_vector(value: np.ndarray | float, n: int, name: str) -> np.ndarray:
    """把标量或一维数组广播为长度 n 的 float64 向量。"""
    if np.isscalar(value):
        return np.full(n, float(value))
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"{name} 必须为标量或长度 {n} 的一维数组，得到 {arr.shape}")
    return arr


@dataclass(frozen=True)
class MarketState:
    """风险中性市场参数。

    Attributes:
        spots: 各标的期初价格，形状 ``(n_assets,)``。
        risk_free_rate: 年化连续复利无风险利率。
        dividend_yields: 各标的连续股息率；可为标量（广播到所有标的）。
        volatilities: 各标的年化波动率；可为标量（广播到所有标的）。
        correlation: 各标的收益率相关系数矩阵，形状
            ``(n_assets, n_assets)``；默认 ``None`` 表示标的相互独立
            （单位阵）。
    """

    spots: np.ndarray
    risk_free_rate: float
    dividend_yields: np.ndarray | float = 0.0
    volatilities: np.ndarray | float = 0.2
    correlation: np.ndarray | None = None

    def __post_init__(self) -> None:
        spots = np.asarray(self.spots, dtype=float)
        if spots.ndim != 1 or spots.size == 0:
            raise ValueError(f"spots 必须是一维非空数组，得到 {spots.shape}")
        if np.any(spots <= 0):
            raise ValueError("spots 必须全部为正")
        object.__setattr__(self, "spots", spots)

        n = spots.size
        object.__setattr__(
            self, "dividend_yields", _as_vector(self.dividend_yields, n, "dividend_yields")
        )
        object.__setattr__(
            self, "volatilities", _as_vector(self.volatilities, n, "volatilities")
        )
        if np.any(self.volatilities < 0):
            raise ValueError("volatilities 必须非负")

        if self.correlation is None:
            corr = np.eye(n)
        else:
            corr = np.asarray(self.correlation, dtype=float)
            if corr.shape != (n, n):
                raise ValueError(
                    f"correlation 形状必须为 {(n, n)}，得到 {corr.shape}"
                )
            if not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1.0):
                raise ValueError("correlation 必须是对称矩阵且对角线为 1")
        object.__setattr__(self, "correlation", corr)

    @property
    def n_assets(self) -> int:
        """标的数量。"""
        return self.spots.size
