"""ELN（反向可转债）定价示例：闭式 vs Monte Carlo + strike 反解。

运行：python examples/eln.py
"""

from __future__ import annotations

import numpy as np

from fistqslab import (
    ELN,
    AnalyticEngine,
    GBMModel,
    MarketState,
    MonteCarloEngine,
    get_eln_strike_from_issue_price,
)


def main() -> None:
    # 市场参数：年化连续复利利率 / 年化波动率 / 时间一律用年
    market = MarketState(
        spots=[100.0],
        risk_free_rate=np.log(1.015),  # 1.5% 年化单利 → 连续复利
        volatilities=0.2287,
    )
    T = 64 / 365  # 64 个自然日

    issue_price = 0.9828
    print(f"目标发行价 issue_price = {issue_price}")

    # 反解公平 strike：使 ELN 理论价格 = 发行价
    strike = get_eln_strike_from_issue_price(issue_price, T, market)
    print(f"反解 strike = {strike:.4f}")

    eln = ELN(strike=strike, maturity_year_fraction=T, issue_price=issue_price)

    # 闭式
    analytic = AnalyticEngine().price(eln, market)
    print(f"闭式价格    = {analytic.price:.6f}")

    # Monte Carlo（100 万路径，内存恒定）
    mc = MonteCarloEngine(n_paths=1_000_000, batch_size=20_000, seed=42).price(
        eln, GBMModel(market)
    )
    print(
        f"MC 价格     = {mc.price:.6f} ± {mc.stderr:.6f}  "
        f"95%CI=[{mc.ci_low:.6f}, {mc.ci_high:.6f}]"
    )
    print(f"MC - 闭式   = {mc.price - analytic.price:+.6f}  "
          f"({3 * mc.stderr:.4f} = 3σ)")


if __name__ == "__main__":
    main()
