"""解析定价引擎：调度产品的闭式解。"""

from __future__ import annotations

from fistqslab.market.state import MarketState
from fistqslab.products.base import Product
from fistqslab.results import PricingResult


class AnalyticEngine:
    """调用 ``product.analytic_price(market)`` 的解析定价引擎。

    产品无闭式解（``analytic_price`` 返回 ``None``）时抛 ``NotImplementedError``。
    """

    def price(self, product: Product, market: MarketState) -> PricingResult:
        """闭式定价。

        Args:
            product: 产品（须实现 ``analytic_price``）。
            market: 市场参数。

        Returns:
            解析定价结果（``method="analytic"``，无 stderr）。
        """
        price = product.analytic_price(market)
        if price is None:
            raise NotImplementedError(
                f"{type(product).__name__} 没有闭式解，请改用 MonteCarloEngine"
            )
        return PricingResult(price=float(price), method="analytic")
