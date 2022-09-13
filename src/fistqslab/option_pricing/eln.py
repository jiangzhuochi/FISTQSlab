from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

from .abc_option import BaseOption
from .util import PriceGenFunc, get_price_path_generator_func


@dataclass
class BaseELN(BaseOption):
    """建仓系列基类
    1. 折价建仓 (Worst of) Equity-Linked Note, (Wo)ELN
    2. 溢价平仓, RELN
    """

    data_path: dict[str, Path]  # key:标的代码, value:标的价格路径数据表的位置
    strike: float  # 行权比率(行权价与初始价的比)
    T: int  # 投资期(以自然日计)

    def __post_init__(self):
        # 各个标的价格路径迭代器函数
        self.S: dict[str, PriceGenFunc] = {}
        for k, path in self.data_path.items():
            self.S[k] = get_price_path_generator_func(path)

        # 各个标的初始价格
        self.S0: dict[str, float] = {}
        for k, gen in self.S.items():
            self.S0[k] = next(gen())[0]

        # 各个标的具体行权价
        self.strike_price: dict[str, float] = {}
        for k, v in self.S0.items():
            self.strike_price[k] = v * self.strike

    @property
    def price(self):

        codes = list(map(lambda it: it[0], self.S.items()))
        gens = list(map(lambda it: it[1](), self.S.items()))
        for path in islice(zip(*gens), 3):
            # path是n元组, n是标的数量
            # 每个元素是一条价格路径
            # TODO: 不需要 df, ndarray 即可
            df = pd.DataFrame(path, index=codes).iloc[:, : self.T + 1]
            self.do_pricing_logic_in_one_path(df)

    @staticmethod
    def do_pricing_logic_in_one_path(df):
        print(df)

    def delta(self):
        pass

    def gamma(self):
        pass

    def vega(self):
        pass

    def theta(self):
        pass
