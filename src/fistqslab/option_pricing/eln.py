from dataclasses import dataclass, field

import numpy as np
from nptyping import Bool, Float64, NDArray, Shape
from scipy.optimize import fsolve

from .mc import MonteCarlo2
from .util import PriceDelta


@dataclass
class BaseELN2(MonteCarlo2):
    """建仓系列基类
    1. 折价建仓 (Worst of) Equity Linked Notes, (Wo)ELN
    2. 溢价平仓, Reverse Equity Linked Notes, RELN
    """

    # 行权比率(行权价与初始价的比)
    strike: float
    # 票据发行价率(发行价格与面值的比)
    issue_price: float
    # 年化息率(根据 T 和 issue_price 计算)
    yield_: float = field(init=False)
    # 无风险收益率(默认值为1年期存款基准利率转化为连续复利收益率)
    r: float = np.log(1 + 0.015)

    def __post_init__(self):

        super().__post_init__()

    @property
    def discount(self):
        """折现因子"""
        return 1 / (1 + self.r) ** (self.T / 365)


@dataclass
class ELN2(BaseELN2):
    """折价建仓"""

    def __post_init__(self):

        super().__post_init__()

        try:
            # 年化息率
            self.yield_ = (1 / self.issue_price - 1) * 365 / self.T
        except Exception:
            pass

    def price(self):
        """向量化定价"""

        # 期末价格, 注意转置, 第 0 维路径数, 第 1 维品种数
        ST_arr: NDArray[Shape["Y, X"], Float64] = self.relative_S[:, :, -1].T
        # 所有的品种都需满足条件
        # 广播运算, ST_arr >= self.strike: Shape["Y, X"]
        # np.all axis=1 最后的维度 (品种) 消失, 变为 Shape["Y,"]
        sig: NDArray[Shape["Y,"], Bool] = np.all(ST_arr >= self.strike, axis=1)
        # 大于等于行权价的情况, Y1 为 sig 中 True 的数量
        upper_scenario: NDArray[Shape["Y1,"], Float64] = np.ones(np.count_nonzero(sig))
        # 表现最差 (即跌幅最大) 的品种 (第 1 维)
        # Y2 为 sig 中 False 的数量, Y1 + Y2 = Y
        # np.min axis=1 最后的维度 (品种) 消失, 变为 Shape["Y2,"]
        # 小于行权价的情况
        lower_scenario: NDArray[Shape["Y2,"], Float64] = np.min(
            ST_arr[~sig] / self.strike, axis=1
        )
        res = np.hstack((upper_scenario, lower_scenario))
        return np.mean(res) * self.discount

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
            strike=self.strike,
            issue_price=self.issue_price,
        ).price()
        delta = None
        if not price_only:
            up_p = type(self)(
                codes=self.codes,
                real_S0=self.real_S0,
                all_relative_S_data=St_paths + epsilon,
                T=left_t,
                strike=self.strike,
                issue_price=self.issue_price,
            ).price()
            lo_p = type(self)(
                codes=self.codes,
                real_S0=self.real_S0,
                all_relative_S_data=St_paths - epsilon,
                T=left_t,
                strike=self.strike,
                issue_price=self.issue_price,
            ).price()
            delta = (up_p - lo_p) / (2 * epsilon)
        return PriceDelta(price, delta)


def get_eln_strike_from_issue_price(
    codes,
    real_S0,
    all_relative_S_data,
    issue_price: float,
    T=64,
):
    """根据 ELN 票据价求行权价"""

    def f(
        strike,
    ):
        op = ELN2(
            codes=codes,
            real_S0=real_S0,
            all_relative_S_data=all_relative_S_data,
            strike=strike,
            issue_price=issue_price,
            T=T,
        )
        return op.price() - issue_price

    return round(fsolve(f, x0=0.95, xtol=1e-5)[0], 4)


@dataclass
class RELN2(BaseELN2):
    """溢价平仓
    注: 在RELN中, issue_price - 1 为区间利息
    """

    # 标的券数量
    number_of_securities: NDArray[Shape["*"], Float64] = field(init=False)

    def __post_init__(self):

        super().__post_init__()

        assert len(self.real_S0) == 1, "目前仅支持单只"
        # 名义本金为 1
        self.number_of_securities = 1 / self.real_S0

        try:
            # 年化息率
            self.yield_ = (self.issue_price - 1) * 365 / self.T
        except Exception:
            pass

    def price(self):
        ST_arr: NDArray[Shape["Y, X"], Float64] = self.relative_S[:, :, -1].T
        # print(ST_arr)
        sig: NDArray[Shape["Y,"], Bool] = np.all(ST_arr > self.strike, axis=1)
        # 大于行权价的情况, Y1 为 sig 中 True 的数量
        # 到期日, 所有标的价格 ST_arr > 行权价 strike, 投资者以行权价卖掉股票, 同时拿到利息
        upper_scenario: NDArray[Shape["Y1,"], Float64] = np.ones(
            np.count_nonzero(sig)
        ) * (self.strike + self.issue_price - 1)
        # 表现最差 (即跌幅最大) 的品种 (第 1 维)
        # Y2 为 sig 中 False 的数量, Y1 + Y2 = Y
        # np.min axis=1 最后的维度 (品种) 消失, 变为 Shape["Y2,"]
        # 小于等于行权价的情况
        # 到期日, 标的价格 ST <= 行权价 strike_price, 投资者拿回股票, 同时拿到利息
        lower_scenario: NDArray[Shape["Y2,"], Float64] = (
            np.min(ST_arr[~sig], axis=1) + self.issue_price - 1
        )
        res = np.hstack((upper_scenario, lower_scenario))
        return np.mean(res) * self.discount

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
            strike=self.strike,
            issue_price=self.issue_price,
        ).price()
        delta = None
        if not price_only:
            up_p = type(self)(
                codes=self.codes,
                real_S0=self.real_S0,
                all_relative_S_data=St_paths + epsilon,
                T=left_t,
                strike=self.strike,
                issue_price=self.issue_price,
            ).price()
            lo_p = type(self)(
                codes=self.codes,
                real_S0=self.real_S0,
                all_relative_S_data=St_paths - epsilon,
                T=left_t,
                strike=self.strike,
                issue_price=self.issue_price,
            ).price()
            delta = (up_p - lo_p) / (2 * epsilon)
        return PriceDelta(price, delta)


def get_reln_issue_price(
    codes,
    real_S0,
    all_relative_S_data,
    strike: float,
    T=64,
):
    """给定行权价, 求issue_price"""

    def f(
        issue_price,
    ):
        op = RELN2(
            codes=codes,
            real_S0=real_S0,
            all_relative_S_data=all_relative_S_data,
            strike=strike,
            issue_price=issue_price,
            T=T,
        )
        return op.price() - 1

    return round(fsolve(f, x0=0.95, xtol=1e-5)[0], 4)
