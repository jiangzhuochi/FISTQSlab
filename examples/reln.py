"""RELN（逆向反向可转债）定价示例。

运行：python examples/reln.py
"""

from __future__ import annotations

import numpy as np

from fistqslab import (
    AnalyticEngine,
    GBMModel,
    MarketState,
    MonteCarloEngine,
    RELN,
    get_reln_issue_price,
)


def main() -> None:
    market = MarketState(
        spots=[100.0],
        risk_free_rate=np.log(1.015),
        volatilities=0.2287,
    )
    T = 64 / 365
    strike = 1.035

    # 平价发行（价格 = 1）对应的发行价
    issue = get_reln_issue_price(strike, T, market)
    print(f"平价发行价 issue_price = {issue:.4f}（strike = {strike}）")

    reln = RELN(strike=strike, issue_price=issue, maturity_year_fraction=T)
    analytic = AnalyticEngine().price(reln, market)
    print(f"闭式价格 = {analytic.price:.6f}")

    mc = MonteCarloEngine(n_paths=500_000, seed=7).price(reln, GBMModel(market))
    print(
        f"MC 价格  = {mc.price:.6f} ± {mc.stderr:.6f}  "
        f"95%CI=[{mc.ci_low:.6f}, {mc.ci_high:.6f}]"
    )


if __name__ == "__main__":
    main()
