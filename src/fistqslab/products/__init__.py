"""产品层：payoff 定义。"""

from fistqslab.products.base import Product
from fistqslab.products.ben import AutoCallNote, BonusEnhancedNote
from fistqslab.products.eln import (
    ELN,
    RELN,
    get_eln_strike_from_issue_price,
    get_reln_issue_price,
)
from fistqslab.products.leverage import LeverageNote
from fistqslab.products.rainbow import RainbowNote
from fistqslab.products.vanilla import (
    DigitalCallOption,
    DigitalPutOption,
    EuropeanCallOption,
    EuropeanPutOption,
)

__all__ = [
    "Product",
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
