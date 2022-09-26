from dataclasses import dataclass, field

import numpy as np
from nptyping import Float64, NDArray, Shape

from .abc_option import BaseOption
from .mc import MonteCarlo, MonteCarlo2
from .util import PriceDelta, get_one_item_dict_kv


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
            ret_pnl[i] = self.do_pricing_logic_in_one_path(i, arr[:, : self.TD + 1])
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
        pnl = (
            self.dividend_rate * self.T / 365 + capital_gains_rate
        ) * self.leverage_multiple

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


@dataclass
class LeverageNote2(MonteCarlo2):

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
        self.issue_price = 1 + self.leverage_cost

    def price(self):

        ST_arr: NDArray[Shape["Y, X"], Float64] = self.relative_S[:, :, -1].T
        pnl = np.mean(
            (self.dividend_rate * self.T / 365 + ST_arr - 1) * self.leverage_multiple
        )
        return self.discount * (1 + pnl)

    def price_and_delta_at(
        self, t: int, St: NDArray[Shape["*"], Float64], price_only=True
    ):
        """计算 delta 值

        dsfParameters
        ----------
        t : int
            时刻, 从 0 到 T 的整数
        St:  NDArray[Shape["X,"], Float64]
            t 时刻标的与real_S0相对价格
        """

        assert len(St) == 1, "目前只支持单只"
        # 剩余自然日
        left_t = self.T - t
        # 剩余交易日
        left_td = left_t * 250 // 365
        # 1 只产品, Y 条路径, left_td + 1 个节点
        left_paths = self.relative_S[:, :, : left_td + 1]
        # 构造从以t时刻价格涨跌为起始的路径, 只支持单只
        St_paths = left_paths * St[0]
        # print(St_paths)
        epsilon = 0.001
        price = type(self)(
            codes=self.codes,
            real_S0=self.real_S0,
            all_relative_S_data=St_paths,
            T=left_t,
            leverage_multiple=self.leverage_multiple,
            dividend_rate=self.dividend_rate,
            leverage_cost=self.leverage_cost,
        ).price()
        delta = None
        if not price_only:
            discount = 1 / (1 + self.r) ** (left_t / 365)
            delta = discount * self.leverage_multiple
        return PriceDelta(price, delta)

    @property
    def discount(self):
        """折现因子"""
        return 1 / (1 + self.r) ** (self.T / 365)
