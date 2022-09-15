from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from operator import ge, gt
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from more_itertools import ilen
from multiprocess import Pool  # type:ignore
from nptyping import Float64, Int, NDArray, Shape

from .abc_option import BaseOption
from .util import (
    PriceGenFunc,
    cmp_dict_all_items,
    get_one_item_dict_kv,
    get_price_path_generator_func,
)


@dataclass
class MonteCarlo:

    # key: 标的代码, value: 标的价格路径数据表的位置
    data_path: dict[str, Path]
    # 模拟路径条数(约定每个标的模拟路径条数相等)
    number_of_paths: int
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
    1. 折价建仓 (Worst of) Equity Linked Notes, (Wo)ELN
    2. 溢价平仓, Reverse Equity Linked Notes, RELN
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
    # 无风险收益率(默认值为1年期存款基准利率转化为连续复利收益率)
    r: float = np.log(1 + 0.015)

    def __post_init__(self):

        super().__post_init__()

        # 各个标的具体行权价
        self.strike_price = {}
        for k, v in self.S0.items():
            self.strike_price[k] = v * self.strike

    @property
    def discount(self):
        """折现因子"""
        return 1 / (1 + self.r) ** (self.T / 365)

    def get_zip_one_path_iterator(
        self,
    ) -> Iterator[tuple[int, NDArray[Shape["A, B"], Float64]]]:
        """将不同底层标的迭代器合并, 返回产出路径编号和路径数组的迭代器"""

        # 所有标的未使用的迭代器
        paths_iters = map(lambda it: it[1](), self._Sitems)
        return map(
            lambda item: (item[0], np.array(item[1])[:, : self.T + 1]),
            # item[0] 是编号
            # item[1] 是 n 元组, n 是标的数量, 每个元素是一条价格路径
            # 转换成 array 后, 每行是一个标的的一条路径, 截取需要的投资期
            enumerate(islice(zip(*paths_iters), self.number_of_paths)),
        )

    def delta(self):
        pass

    def gamma(self):
        pass

    def vega(self):
        pass

    def theta(self):
        pass


# https://www.dbs.com.sg/treasures/investments/product-suite/equities/equity-linked-investments
# https://www.dfzq.com.hk/main/mainbusiness/wealthmanagement/structured-products/len/index.shtml
@dataclass
class ELN(BaseELN):
    """折价建仓"""

    # 名义本金(票据面值)
    nominal_amount: float = 1e6

    def __post_init__(self):

        super().__post_init__()

        # 年化息率
        self.yield_ = (1 / self.issue_price - 1) * 365 / self.T

    @property
    def price(self):

        ret_sum = 0
        for i, arr in self.get_zip_one_path_iterator():
            ret_sum += self.do_pricing_logic_in_one_path(i, arr)

        return ret_sum / self.number_of_paths / self.nominal_amount

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


# 到期的具体损益不明确, 待进一步确认
# https://www.dbs.com.sg/treasures/investments/product-suite/structured-investments/reverse-equity-linked-notes
@dataclass
class RELN(BaseELN):
    """折价建仓"""

    # key: 标的代码, value: 标的数量
    number_of_securities: dict[str, int] = field(kw_only=True)
    # 名义本金(根据起始日股票市场价值计算)
    nominal_amount: float = field(init=False)

    def __post_init__(self):

        super().__post_init__()

        assert self.S0.keys() == self.number_of_securities.keys()
        # 名义本金, 目前只支持单只标的
        if len(self.S0) == 1:
            _, s = get_one_item_dict_kv(self.S0)
            _, n = get_one_item_dict_kv(self.number_of_securities)
            self.nominal_amount = s * n
        else:
            raise NotImplemented

        # 年化息率
        self.yield_ = (self.issue_price - 1) * 365 / self.T

    @property
    def price(self):

        ret_sum = 0
        for i, arr in self.get_zip_one_path_iterator():
            ret_sum += self.do_pricing_logic_in_one_path(i, arr)

        return ret_sum / self.number_of_paths / self.nominal_amount

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
        if cmp_dict_all_items(ST, self.strike_price, gt):
            # print(">", ST, self.strike_price)
            # print("+=========", self.scenario_ST_gt_strike_pnl)
            return self.scenario_ST_gt_strike_pnl
        else:
            # print("<=", ST, self.strike_price)
            # print(self.scenario_ST_le_strike_pnl(ST))
            return self.scenario_ST_le_strike_pnl(ST)

    @cached_property
    def scenario_ST_gt_strike_pnl(self):
        """到期日, 标的价格 ST > 行权价 strike_price, 投资者以行权价卖掉股票, 同时拿到利息"""

        return self.discount * (
            self.nominal_amount * self.strike
            + self.nominal_amount * (self.issue_price - 1)
        )

    def scenario_ST_le_strike_pnl(self, ST):
        """到期日, 标的价格 ST <= 行权价 strike_price, 投资者拿回股票, 同时拿到利息"""

        _, s = get_one_item_dict_kv(ST)
        _, n = get_one_item_dict_kv(self.number_of_securities)

        return self.discount * (s * n + self.nominal_amount * (self.issue_price - 1))
