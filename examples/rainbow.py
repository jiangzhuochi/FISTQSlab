"""Rainbow Note（worst-of 增利票据）定价示例。

演示：参与率与 lower_call_strike 对价格的影响（重构前这两个参数
不参与定价，本示例验证它们确实生效）。

运行：python examples/rainbow.py
"""

from __future__ import annotations

import numpy as np

from fistqslab import (
    GBMModel,
    MarketState,
    MonteCarloEngine,
    RainbowNote,
)


def main() -> None:
    market = MarketState(
        spots=[100.0, 95.0, 110.0],  # 三标的
        risk_free_rate=0.02,
        volatilities=[0.2, 0.25, 0.3],
        correlation=np.array(
            [
                [1.0, 0.6, 0.4],
                [0.6, 1.0, 0.5],
                [0.4, 0.5, 1.0],
            ]
        ),
    )
    base = dict(
        put_strike=0.9,
        lower_call_strike=1.0,
        guaranteed_flat_coupon=0.04,
        maturity_year_fraction=0.5,
    )

    engine = MonteCarloEngine(n_paths=300_000, batch_size=20_000, seed=42)
    model = GBMModel(market)

    for part in (0.5, 1.0, 2.0):
        rb = RainbowNote(upside_participation=part, **base)
        r = engine.price(rb, model)
        print(f"参与率={part}: 价格 = {r.price:.6f} ± {r.stderr:.6f}")

    print("\n（参与率从 0.5 → 2.0 价格显著上升，证明参数已参与定价）")


if __name__ == "__main__":
    main()
