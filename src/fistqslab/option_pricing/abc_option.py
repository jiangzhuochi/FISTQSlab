from abc import ABC, abstractmethod
from typing import Literal

common_field = ["price", "delta", "gamma", "theta", "vega"]
# Unpack operator in subscript requires Python 3.11 or newer
# FIELD = Literal[*common_field]
FIELD = Literal["price", "delta", "gamma", "theta", "vega"]


class OptionType(ABC):
    """期权种类: 看涨(认购)或看跌(认沽)"""

    pass


class Call(OptionType):
    pass


class Put(OptionType):
    pass


class BaseOption(ABC):
    """抽象类, 期权"""

    @property
    def price(self):
        raise NotImplemented

    @property
    def delta(self):
        raise NotImplemented

    @property
    def gamma(self):
        raise NotImplemented

    @property
    def theta(self):
        raise NotImplemented

    @property
    def vega(self):
        raise NotImplemented
