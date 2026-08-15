"""Leverage Note（杠杆票据）定价示例 + Greeks。

运行：python examples/leverage.py
"""

from __future__ import annotations

import numpy as np

from fistqslab import (
    AnalyticEngine,
    GBMModel,
    GreeksCalculator,
    LeverageNote,
    MarketState,
    MonteCarloEngine,
)


def main() -> None:
    market = MarketState(spots=[100.0], risk_free_rate=0.02, volatilities=0.25)
    T = 0.5
    lev = LeverageNote(
        leverage_multiple=2.0,
        maturity_year_fraction=T,
        dividend_rate=0.04,
    )

    analytic = AnalyticEngine().price(lev, market)
    mc = MonteCarloEngine(n_paths=500_000, seed=42).price(lev, GBMModel(market))
    print(f"闭式价格 = {analytic.price:.6f}   MC = {mc.price:.6f} ± {mc.stderr:.6f}")

    # Greeks（有限差分 + CRN）
    g = GreeksCalculator(MonteCarloEngine(n_paths=200_000, seed=7))
    delta = g.delta(lev, GBMModel(market))
    print(f"delta = {delta[0]:.4f}（payoff 只依赖相对收益，对绝对 spot 不敏感）")


if __name__ == "__main__":
    main()
