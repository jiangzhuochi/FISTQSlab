from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from operator import ge, gt
from typing import cast

import numpy as np
import pandas as pd
from nptyping import Float64, NDArray, Shape

from .abc_option import BaseOption
from .mc import MonteCarlo
from .util import cmp_dict_all_items, get_one_item_dict_kv


@dataclass
class BaseBEN(BaseOption, MonteCarlo):

    # 票据期限(以自然日计)
    T: int
    # 票据期限(以交易日计)
    TD: int = field(init=False)
    # 下界
    put_strike: float
    # 上界
    coupon_barrier: float
    # 票息
    bonus_coupon: float
    # 最低赎回比例 Minimum Redemption Amount
    min_redemption: float | None = None
    # 无风险收益率(默认值为1年期存款基准利率转化为连续复利收益率)
    r: float = np.log(1 + 0.015)
    # 名义本金(票据面值)
    nominal_amount: float = 1e6

    def __post_init__(self):

        super().__post_init__()

        self.TD = self.T * 250 // 365

    @property
    def price(self):
        ret_pnl = np.empty(self.number_of_paths, dtype=float)
        for i, arr in self.get_zip_one_path_iterator():
            ret_pnl[i] = self.do_pricing_logic_in_one_path(i, arr[:, : self.TD + 1])
        return np.mean(ret_pnl, axis=0) / self.nominal_amount

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
        code, worst_pct_chg = self.find_worst_target(ST)
        if 1 + worst_pct_chg >= self.coupon_barrier:
            fv = self.nominal_amount * (1 + max(self.bonus_coupon, worst_pct_chg))
        elif (
            1 + worst_pct_chg < self.coupon_barrier
            and 1 + worst_pct_chg >= self.put_strike
        ):
            fv = self.nominal_amount
        else:
            if self.min_redemption is None:
                # 按照行权价转换为表现最差的股票
                fv = self.nominal_amount // (self.put_strike * self.S0[code]) * ST[code]
            else:
                # 如果给定最低赎回比例, 则下有底, 有保护
                fv = max(
                    self.nominal_amount * self.min_redemption,
                    self.nominal_amount // (self.put_strike * self.S0[code]) * ST[code],
                )
        return self.discount * fv

    def find_worst_target(self, ST) -> tuple[str, float]:
        """找出表现最差的标的, 返回其代码和涨跌幅"""

        # 所有标的涨跌幅
        pct_chts = {}
        for key in self.S0:
            pct_chts[key] = (ST[key] - self.S0[key]) / self.S0[key]
        # 升序排列, 第一个是表现最差的
        return next(iter(sorted(pct_chts.items(), key=lambda x: x[1])))

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


@dataclass
class AutoCallBEN(BaseBEN):

    # 敲入价格
    ki_barrier: float = field(kw_only=True)
    # 敲出价格
    autocall_barrier: float = field(kw_only=True)
    # 观察频率(以天计)
    # TODO: 通过观察频率计算或者直接给出一个观察日列表, 特别注意交易日和自然日的对应
    autocall_frequency: int = field(kw_only=True)
