"""几何布朗运动（GBM）模拟。

核心设计原则（对应重构前 ``get_stock_path`` 的内存问题）：

- **分块（chunked）**：以生成器产出固定大小批次，内存占用与
  ``batch_size`` 成正比，与总路径数无关。默认参数下 100 万条路径
  内存也恒定在几十 MB 量级（旧实现单数组峰值可达 ~96 GB）。
- **terminal-only**：绝大多数票据（ELN/BEN/Rainbow/Leverage）的 payoff
  只看终值，直接模拟 ``S_T`` 而非整条路径：
  ``S_T = S0·exp((r-q-σ²/2)T + σ√T·Z)``，内存 O(路径数) 而非
  O(路径数 × 时间步)。
- **相关性**：多标的通过 Cholesky 分解 ``L·Lᵀ = corr`` 施加相关性。
- **可复现**：一律 ``np.random.default_rng(seed)``；相同 ``seed`` 得到
  逐位一致的模拟，这是 Common Random Numbers（CRN）Greeks 的基础。
- **对偶变量（antithetic）**：``n_paths`` 为独立抽样数，有效路径数为
  ``2 × n_paths``。

返回的均为**相对价格** ``S/S0``（形状 ``(n_assets, ...)``），产品 payoff
直接使用相对收益（worst-of 等），无需再除以期初价。
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from fistqslab.market.state import MarketState

__all__ = ["GBMModel"]


class GBMModel:
    """多标的几何布朗运动风险中性模型。

    Args:
        market: 风险中性市场参数（含相关性）。
    """

    def __init__(self, market: MarketState):
        self.market = market
        # 漂移向量：r - q - σ²/2（逐资产）
        self._drift = market.risk_free_rate - market.dividend_yields - 0.5 * market.volatilities**2
        self._sigma = market.volatilities
        self._chol = np.linalg.cholesky(market.correlation)

    @property
    def n_assets(self) -> int:
        return self.market.n_assets

    def _make_rng(self, seed: int | None) -> np.random.Generator:
        return np.random.default_rng(seed)

    def terminal_batches(
        self,
        T: float,
        n_paths: int,
        batch_size: int = 20_000,
        antithetic: bool = True,
        seed: int | None = None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """按批次产出终值相对价格。

        Args:
            T: 到期年数（year fraction）。
            n_paths: 独立随机抽样数（antithetic 时有效路径数为 2×）。
            batch_size: 每个批次的独立抽样数（内存上界）。
            antithetic: 是否使用对偶变量。
            seed: 随机种子；相同 seed 产出完全一致的结果。

        Yields:
            ``(rel, n_eff)``：``rel`` 形状 ``(n_assets, n_eff)``，
            ``n_eff`` 为本批有效路径数。
        """
        if T <= 0:
            raise ValueError("T 必须为正")
        rng = self._make_rng(seed)
        drift_t = self._drift * T
        diff_t = self._sigma * np.sqrt(T)

        remaining = n_paths
        while remaining > 0:
            n = min(batch_size, remaining)
            z = rng.standard_normal((self.n_assets, n))
            if antithetic:
                z = np.hstack((z, -z))
            z_corr = self._chol @ z
            rel = np.exp(drift_t[:, None] + diff_t[:, None] * z_corr)
            yield rel, rel.shape[1]
            remaining -= n

    def path_batches(
        self,
        obs_year_fractions: np.ndarray,
        n_paths: int,
        batch_size: int = 20_000,
        antithetic: bool = True,
        seed: int | None = None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """按批次产出观察日路径相对价格。

        只模拟 ``obs_year_fractions`` 上的值（而非逐日节点），增量用
        Brownian 增量叠加，适用于 AutoCall 等观察日驱动的产品。

        Args:
            obs_year_fractions: 观察日年数（升序，含到期日）。
            n_paths: 独立随机抽样数。
            batch_size: 每批独立抽样数。
            antithetic: 是否使用对偶变量。
            seed: 随机种子。

        Yields:
            ``(rel, n_eff)``：``rel`` 形状
            ``(n_assets, n_obs, n_eff)``，``rel[i, k, j]`` 为标的 i 在
            第 k 个观察日、路径 j 的相对价格。
        """
        obs = np.asarray(obs_year_fractions, dtype=float)
        if obs.ndim != 1 or obs.size == 0:
            raise ValueError("obs_year_fractions 必须是一维非空数组")
        if np.any(np.diff(obs) <= 0) or obs[0] <= 0:
            raise ValueError("obs_year_fractions 必须严格递增且全部为正")
        rng = self._make_rng(seed)
        n_obs = obs.size
        sqrt_deltas = np.sqrt(np.diff(np.concatenate(([0.0], obs))))

        remaining = n_paths
        while remaining > 0:
            n = min(batch_size, remaining)
            z = rng.standard_normal((n_obs, self.n_assets, n))
            if antithetic:
                z = np.concatenate((z, -z), axis=-1)
            # 相关增量：(n_obs, n_assets, n_eff)
            z_corr = np.einsum("ij,kjl->kil", self._chol, z)
            # 相关 BM 在观察日的累计值
            w = np.cumsum(sqrt_deltas[:, None, None] * z_corr, axis=0)
            rel = np.exp(
                self._drift[None, :, None] * obs[:, None, None]
                + self._sigma[None, :, None] * w
            )
            # 转置为 (n_assets, n_obs, n_eff)
            rel = np.transpose(rel, (1, 0, 2))
            yield rel, rel.shape[-1]
            remaining -= n
