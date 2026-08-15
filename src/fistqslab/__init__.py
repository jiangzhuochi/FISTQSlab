"""FISTQSlab：小型结构化衍生品定价库。

三层架构：**产品（payoff）— 模型（风险中性过程）— 引擎（计算）**，
市场参数统一由 :class:`~fistqslab.market.state.MarketState` 提供。

快速开始::

    from fistqslab import (
        MarketState, GBMModel, MonteCarloEngine, AnalyticEngine, ELN,
    )

    market = MarketState(spots=[100.0], risk_free_rate=0.02, volatilities=0.2)
    eln = ELN(strike=0.95, maturity_year_fraction=0.5)
    model = GBMModel(market)
    engine = MonteCarloEngine(n_paths=100_000, seed=42)
    result = engine.price(eln, model)   # result.price / result.stderr / result.ci_*
"""

from fistqslab.engines import AnalyticEngine, MonteCarloEngine
from fistqslab.market import MarketState, year_fraction
from fistqslab.models import (
    GBMModel,
    bs_call,
    bs_digital_call,
    bs_digital_put,
    bs_greeks,
    bs_put,
)
from fistqslab.products import (
    ELN,
    RELN,
    AutoCallNote,
    BonusEnhancedNote,
    DigitalCallOption,
    DigitalPutOption,
    EuropeanCallOption,
    EuropeanPutOption,
    LeverageNote,
    RainbowNote,
    get_eln_strike_from_issue_price,
    get_reln_issue_price,
)
from fistqslab.results import PricingResult
from fistqslab.risk import GreeksCalculator

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "MarketState",
    "year_fraction",
    "GBMModel",
    "bs_call",
    "bs_put",
    "bs_digital_call",
    "bs_digital_put",
    "bs_greeks",
    "AnalyticEngine",
    "MonteCarloEngine",
    "PricingResult",
    "GreeksCalculator",
    "EuropeanCallOption",
    "EuropeanPutOption",
    "DigitalCallOption",
    "DigitalPutOption",
    "ELN",
    "RELN",
    "get_eln_strike_from_issue_price",
    "get_reln_issue_price",
    "BonusEnhancedNote",
    "AutoCallNote",
    "RainbowNote",
    "LeverageNote",
]
