from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from operator import ge, gt

import numpy as np
from nptyping import Float64, NDArray, Shape

from .abc_option import BaseOption
from .mc import MonteCarlo
from .util import cmp_dict_all_items, get_one_item_dict_kv


@dataclass
class LeverageNote(BaseOption, MonteCarlo):

    # 票据期限(以自然日计)
    T: int
    # 票据期限(以交易日计)
    TD: int = field(init=False)
    # 杠杆倍数
    leverage_multiple: float
    # 股息率(年化)
    dividend_rate: float
    # 杠杆成本
    leverage_cost: float
    # 票据发行价
    issue_price: float = field(init=False)
    # 无风险收益率(默认值为1年期存款基准利率转化为连续复利收益率)
    r: float = np.log(1 + 0.015)

    def __post_init__(self):

        super().__post_init__()

        self.TD = self.T * 250 // 365
        self.issue_price = 1 + self.leverage_cost

    @property
    def price(self):
        ret_pnl = np.empty(self.number_of_paths, dtype=float)
        for i, arr in self.get_zip_one_path_iterator():
            ret_pnl[i] = self.do_pricing_logic_in_one_path(i, arr[:, : self.T + 1])
        return np.mean(ret_pnl, axis=0)

    def do_pricing_logic_in_one_path(
        self, i: int, arr: NDArray[Shape["A, B"], Float64]
    ) -> float:
        """分析一条路径, 返回该路径的损益情况

        Parameters
        ----------
        i : int
            路径序号
        arr : NDArray[Shape[&quot;A, B&quot;], Float64]
            shape 为 (A, B) 的数组, A = len(self.codes), B = self.TD + 1
        """

        # 该路径的期末价格
        ST: dict[str, float] = dict(zip(self.codes, arr[:, -1]))
        # 票据资本利得部分收益
        _, s0 = get_one_item_dict_kv(self.S0)
        _, st = get_one_item_dict_kv(ST)
        capital_gains_rate = (st - s0) / s0
        # 总收益率
        pnl = (self.dividend_rate + capital_gains_rate) * self.leverage_multiple

        return self.discount * (1 + pnl)

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
