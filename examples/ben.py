"""BEN（增利票据）定价示例：保本 / 非保本 + 票息敏感性。

运行：python examples/ben.py
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fistqslab import (
    AnalyticEngine,
    BonusEnhancedNote,
    GBMModel,
    MarketState,
    MonteCarloEngine,
)


def main() -> None:
    market = MarketState(spots=[100.0], risk_free_rate=0.02, volatilities=0.25)
    T = 183 / 365
    base = dict(
        put_strike=0.9,
        coupon_barrier=1.0,
        maturity_year_fraction=T,
    )
    ben = BonusEnhancedNote(
        bonus_coupon=0.065 * 183 / 365,  # 整个期限总票息
        min_redemption=0.85,
        **base,
    )
    ben_no_mr = BonusEnhancedNote(bonus_coupon=0.065 * 183 / 365, **base)

    model = GBMModel(market)
    engine = MonteCarloEngine(n_paths=200_000, batch_size=20_000, seed=42)
    analytic = AnalyticEngine()

    for name, p in [("保本(85%)", ben), ("无保本", ben_no_mr)]:
        r_an = analytic.price(p, market)
        r_mc = engine.price(p, model)
        print(f"{name}: 闭式 = {r_an.price:.6f}  MC = {r_mc.price:.6f} ± {r_mc.stderr:.6f}")

    # 票息 vs 价格（解析）
    coupons = np.linspace(0.0, 0.12, 61)
    prices = [
        BonusEnhancedNote(bonus_coupon=c, **base).analytic_price(market)
        for c in coupons
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coupons * 365 / 183 * 100, prices)
    ax.set_xlabel("票息（年化 %）")
    ax.set_ylabel("价格（单位本金）")
    ax.set_title("BEN 价格 vs 票息（闭式）")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("examples/img/ben_coupon_sensitivity.png", dpi=120)
    print("已保存 examples/img/ben_coupon_sensitivity.png")


if __name__ == "__main__":
    main()
