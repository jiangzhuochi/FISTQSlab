from dataclasses import dataclass, field
from itertools import islice
from operator import ge, lt
from pathlib import Path

import numpy as np
import pandas as pd
from nptyping import Float64, Int, NDArray, Shape

from .abc_option import BaseOption
from .util import PriceGenFunc, cmp_dict_all_items, get_price_path_generator_func


@dataclass
class MonteCarlo:

    # key: 标的代码, value: 标的价格路径数据表的位置
    data_path: dict[str, Path]
    # key: 标的代码, value: 标的价格路径迭代器函数
    S: dict[str, PriceGenFunc] = field(init=False)
    # key: 标的代码, value: 标的初始价格
    S0: dict[str, float] = field(init=False)
    # 标的代码列表
    codes: list[str] = field(init=False)

    def __post_init__(self):
        # 各个标的价格路径迭代器函数
        self.S = {}
        for k, path in self.data_path.items():
            self.S[k] = get_price_path_generator_func(path)

        # 各个标的初始价格
        self.S0 = {}
        for k, gen in self.S.items():
            self.S0[k] = next(gen())[0]

        # 可遍历的 dict_items
        self._Sitems = self.S.items()

        # 标的资产代码列表
        self.codes = list(map(lambda it: it[0], self._Sitems))


@dataclass
class BaseELN(BaseOption, MonteCarlo):
    """建仓系列基类
    1. 折价建仓 (Worst of) Equity-Linked Note, (Wo)ELN
    2. 溢价平仓, RELN
    """

    # 投资期(以自然日计)
    T: int
    # 行权比率(行权价与初始价的比)
    strike: float
    # key: 标的代码, value: 标的具体行权价(元, 根据 S0 和 strike 计算)
    strike_price: dict[str, float] = field(init=False)
    # 票据发行价率(发行价格与面值的比)
    issue_price: float
    # 年化息率(根据 T 和 issue_price 计算)
    yield_: float = field(init=False)
    # 名义金额(票据面值)
    nominal_amount: float = 1e6
    # 无风险收益率(默认值为1年期存款基准利率转化为连续复利收益率)
    r: float = np.log(1 + 0.015)

    def __post_init__(self):

        super().__post_init__()

        # 各个标的具体行权价
        self.strike_price = {}
        for k, v in self.S0.items():
            self.strike_price[k] = v * self.strike

        # 年化息率
        self.yield_ = (1 / self.issue_price - 1) * 365 / self.T

    @property
    def price(self):

        gens = list(map(lambda it: it[1](), self._Sitems))
        number_of_paths = 20000
        ret_sum = 0
        for i, price_path in enumerate(islice(zip(*gens), number_of_paths)):
            # price_path 是 n 元组, n 是标的数量, 每个元素是一条价格路径
            arr = np.array(price_path)[:, : self.T + 1]
            ret_sum += self.do_pricing_logic_in_one_path(i, arr)
        return ret_sum / number_of_paths / self.nominal_amount

    def do_pricing_logic_in_one_path(
        self, i, arr: NDArray[Shape["A, B"], Float64]
    ) -> float:
        """_summary_

        Parameters
        ----------
        i : _type_
            _description_
        arr : NDArray[Shape[&quot;A, B&quot;], Float64]
            shape 为 (A, B) 的数组, A = len(self.codes), B = self.T + 1
        """

        # print("{:-^20}".format(f" path {i} "))
        assert arr.shape == (len(self.codes), self.T + 1)
        # 该路径的期末价格
        ST: dict[str, float] = dict(zip(self.codes, arr[:, -1]))
        if cmp_dict_all_items(ST, self.strike_price, ge):
            # Scenario a
            # 到期日, 标的价格 ST >= 行权价 strike_price, 投资者拿回 nominal_amount
            # print(">=", ST, self.strike_price)
            return self.discount * self.nominal_amount
        else:
            # Scenario b
            # 到期日, 标的价格 ST < 行权价 strike_price, 投资者拿回 ST * (nominal_amount // strike_price)
            # print("<", ST, self.strike_price)
            return self.discount * self.find_trigger_value(ST)

    def find_trigger_value(self, ST: dict[str, float]) -> float:
        """当触发行权价时, 投资者拿到的金额
        目前只写单一标的
        """
        if len(ST) == 1:
            key = next(iter(ST.keys()))
            return ST[key] * (self.nominal_amount // self.strike_price[key])
        else:
            raise NotImplemented

    @property
    def discount(self):
        """折现因子"""
        return 1 / (1 + self.r) ** (self.T / 365)

    def delta(self):
        pass

    def gamma(self):
        pass

    def vega(self):
        pass

    def theta(self):
        pass
