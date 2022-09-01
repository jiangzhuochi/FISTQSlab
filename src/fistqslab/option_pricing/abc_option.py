from abc import ABC, abstractmethod


class OptionType:
    """期权种类: 看涨(认购)或看跌(认沽)"""

    pass


class Call(OptionType):
    pass


class Put(OptionType):
    pass


class BaseOption(ABC):
    """抽象类, 期权"""

    @property
    @abstractmethod
    def price(self):
        raise NotImplemented

    @property
    @abstractmethod
    def delta(self):
        raise NotImplemented

    @property
    @abstractmethod
    def gamma(self):
        raise NotImplemented

    @property
    @abstractmethod
    def theta(self):
        raise NotImplemented

    @property
    @abstractmethod
    def vega(self):
        raise NotImplemented
