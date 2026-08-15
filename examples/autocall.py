"""AutoCall（自动赎回票据）定价示例。

演示：观察日生成、路径模拟（只在观察日布点，而非逐日）、
提前赎回现金流现值。

运行：python examples/autocall.py
"""

from __future__ import annotations

import numpy as np

from fistqslab import (
    AutoCallNote,
    GBMModel,
    MarketState,
    MonteCarloEngine,
)


def main() -> None:
    market = MarketState(spots=[100.0], risk_free_rate=0.02, volatilities=0.25)
    ac = AutoCallNote(
        ki_barrier=0.7,          # 敲入价（到期失效）
        autocall_barrier=1.0,    # 敲出价
        observation_frequency_days=30,  # 每月观察
        bonus_coupon=0.08,       # 整个期限总票息
        maturity_year_fraction=1.0,
        min_redemption=0.85,
    )
    obs_days = np.round(ac.observation_year_fractions * 365, 1)
    print(f"观察日（自然日）：{obs_days}")

    model = GBMModel(market)
    engine = MonteCarloEngine(n_paths=500_000, batch_size=20_000, seed=42)
    r = engine.price(ac, model)
    print(
        f"AutoCall 价格 = {r.price:.6f} ± {r.stderr:.6f}  "
        f"95%CI=[{r.ci_low:.6f}, {r.ci_high:.6f}]"
    )
    print(f"有效路径数 = {r.n_effective_paths}（内存与路径数无关，仅与 batch 相关）")


if __name__ == "__main__":
    main()
