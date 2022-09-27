from dataclasses import dataclass, field

import numpy as np
from nptyping import Float64, NDArray, Shape

from .abc_option import BaseOption
from .mc import MonteCarlo, MonteCarlo2
from .util import PriceDelta, find_worst_target


@dataclass
class RainbowNote(BaseOption, MonteCarlo):

    # 票据期限(以自然日计)
    T: int
    # 票据期限(以交易日计)
    TD: int = field(init=False)
    # 下界
    put_strike: float
    # 低上界
    lower_call_strike: float
    # 上方参与率
    upside_participation: float
    # 票息(年化)
    guaranteed_flat_coupon: float
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
        code, worst_pct_chg = find_worst_target(self.S0, ST)
        real_rate_of_return = self.guaranteed_flat_coupon * self.T / 365
        if worst_pct_chg >= 0:
            # 本金 + 固定票息收益 + 表现最差标的的涨幅
            fv = self.nominal_amount * (1 + real_rate_of_return + worst_pct_chg)
        elif worst_pct_chg < 0 and 1 + worst_pct_chg >= self.put_strike:
            # 本金 + 固定票息收益
            fv = self.nominal_amount * (1 + real_rate_of_return)
        else:
            # 固定票息收益 + 本金按照行权价转股为表现最差的标的股票
            fv = (
                self.nominal_amount * real_rate_of_return
                + self.nominal_amount // (self.put_strike * self.S0[code]) * ST[code]
            )

        return self.discount * fv

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
class RainbowNote2(MonteCarlo2):

    # 下界
    put_strike: float
    # 低上界
    lower_call_strike: float
    # 上方参与率
    upside_participation: float
    # 票息(区间收益)
    guaranteed_flat_coupon: float
    # 无风险收益率(默认值为1年期存款基准利率转化为连续复利收益率)
    r: float = np.log(1 + 0.015)

    def __post_init__(self):

        super().__post_init__()
        self.TD = self.T * 250 // 365

    def price(self):

        # 期末价格, 注意转置, 第 0 维路径数, 第 1 维品种数
        ST_arr: NDArray[Shape["Y, X"], Float64] = self.relative_S[:, :, -1].T
        worst_pnl = np.sort(ST_arr)[:, 0]

        # 最差标的比 coupon_barrier 还要高
        s1 = worst_pnl[worst_pnl >= 1] + self.guaranteed_flat_coupon 
        # 最差标的介于 coupon_barrier 和 put_strike 之间
        s2 = np.ones(
            np.count_nonzero((worst_pnl >= self.put_strike) & (worst_pnl < 1))
        ) * (1 + self.guaranteed_flat_coupon )
        # 最差标的低于 put_strike
        s3 = (
            self.guaranteed_flat_coupon 
            + worst_pnl[worst_pnl < self.put_strike] / self.put_strike
        )

        return self.discount * np.mean(np.hstack((s1, s2, s3)))

    def price_and_delta_at(
        self, t: int, St: NDArray[Shape["*"], Float64], underlying: int, price_only=True
    ):
        """计算 delta 值

        dsfParameters
        ----------
        t : int
            时刻, 从 0 到 T 的整数
        underlying:  int
            求哪一个资产的偏导数
        St:  NDArray[Shape["X,"], Float64]
            t 时刻标的与real_S0相对价格
        """

        assert len(St) == len(self.codes), f"标的数和价格数不一致"
        # 剩余自然日
        left_t = self.T - t
        # 剩余交易日
        left_td = left_t * 250 // 365
        # X 只产品, Y 条路径, left_td + 1 个节点, 注意 copy
        left_paths = self.relative_S[:, :, : left_td + 1].copy()
        # 构造所有标的t时刻价格涨跌为起始的路径
        for i, st in enumerate(St):
            left_paths[i] = left_paths[i] * st
        price = type(self)(
            codes=self.codes,
            real_S0=self.real_S0,
            all_relative_S_data=left_paths,
            T=left_t,
            put_strike=self.put_strike,
            lower_call_strike=self.lower_call_strike,
            upside_participation=self.upside_participation,
            guaranteed_flat_coupon=self.guaranteed_flat_coupon,
        ).price()
        delta = None
        if not price_only:
            epsilon = 0.001
            upper = left_paths.copy()
            upper[underlying] += epsilon
            lower = left_paths.copy()
            lower[underlying] -= epsilon
            up_p = type(self)(
                codes=self.codes,
                real_S0=self.real_S0,
                all_relative_S_data=upper,
                T=left_t,
                put_strike=self.put_strike,
                lower_call_strike=self.lower_call_strike,
                upside_participation=self.upside_participation,
                guaranteed_flat_coupon=self.guaranteed_flat_coupon,
            ).price()
            lo_p = type(self)(
                codes=self.codes,
                real_S0=self.real_S0,
                all_relative_S_data=lower,
                T=left_t,
                put_strike=self.put_strike,
                lower_call_strike=self.lower_call_strike,
                upside_participation=self.upside_participation,
                guaranteed_flat_coupon=self.guaranteed_flat_coupon,
            ).price()
            delta = (up_p - lo_p) / (2 * epsilon)
        return PriceDelta(price, delta)

    @property
    def discount(self):
        """折现因子"""
        return 1 / (1 + self.r) ** (self.T / 365)
