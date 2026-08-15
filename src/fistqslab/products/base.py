"""产品（payoff）抽象基类。

设计原则（对应重构建议"产品—模型—引擎"三层分离）：

- **产品只定义 payoff**，不持有市场参数、不生成随机数；
- 引擎负责计算（Monte Carlo 循环 / 解析调度）；
- 市场参数（spot/vol/rate/correlation）统一由 ``MarketState`` 提供。

产品内部一律按**单位名义本金**计价（1 元本金），payoff 返回每路径的
收益（单位本金），折现由引擎统一处理（``payoff_terminal`` 返回未折现
收益；``payoff_paths`` 因观察日不同需各自折现，由产品自行折现）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from fistqslab.market.state import MarketState


class Product(ABC):
    """衍生品产品抽象基类。

    子类需实现：

    - ``maturity_year_fraction``：到期年数；
    - ``observation_year_fractions``（可选，默认 ``None`` 表示仅看终值）：
      观察日年数数组，非 ``None`` 时产品为路径依赖（如 AutoCall），
      必须实现 ``payoff_paths``；
    - ``payoff_terminal`` 与/或 ``payoff_paths``；
    - ``analytic_price``（可选）：闭式解，供 ``AnalyticEngine`` 与
      Monte Carlo 对照使用。
    """

    @property
    @abstractmethod
    def maturity_year_fraction(self) -> float:
        """到期年数（year fraction）。"""

    @property
    def observation_year_fractions(self) -> np.ndarray | None:
        """观察日年数；``None`` 表示 payoff 只依赖终值。"""
        return None

    @abstractmethod
    def payoff_terminal(self, rel: np.ndarray, spots: np.ndarray) -> np.ndarray:
        """计算每路径的到期收益（未折现，单位名义本金）。

        Args:
            rel: 终值相对价格（相对期初 spot），形状
                ``(n_assets, n_paths)``。
            spots: 各标的价格，形状 ``(n_assets,)``；按金额计价的产品
                （如 vanilla 期权）用 ``rel * spots`` 还原绝对价格。

        Returns:
            每路径收益，形状 ``(n_paths,)``。
        """

    def payoff_paths(
        self, rel: np.ndarray, spots: np.ndarray, market: MarketState
    ) -> np.ndarray:
        """计算每路径的收益现值（已按观察日折现，单位名义本金）。

        Args:
            rel: 观察日相对价格，形状 ``(n_assets, n_obs, n_paths)``。
            spots: 各标的价格，形状 ``(n_assets,)``。
            market: 市场参数（观察日折现需要利率）。

        Returns:
            每路径收益现值，形状 ``(n_paths,)``。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 未实现 payoff_paths（路径依赖产品必需）"
        )

    def analytic_price(self, market: MarketState) -> float | None:
        """可选：闭式解价格（单位本金现值）。

        返回 ``None`` 表示无闭式解。
        """
        return None
