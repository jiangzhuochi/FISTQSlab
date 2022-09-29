from dataclasses import dataclass, field

import numpy as np
from nptyping import Float64, NDArray, Shape

from .abc_option import BaseOption
from .mc import MonteCarlo, MonteCarlo2
from .util import PriceDelta, find_worst_target


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
        code, worst_pct_chg = find_worst_target(self.S0, ST)
        if 1 + worst_pct_chg >= self.coupon_barrier:
            fv = self.nominal_amount * (
                1 + max(self.bonus_coupon * self.T / 365, worst_pct_chg)
            )
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


@dataclass
class BaseBEN2(MonteCarlo2):

    # 下界
    put_strike: float
    # 上界
    coupon_barrier: float
    # 票息(区间收益)
    bonus_coupon: float
    # 最低赎回比例 Minimum Redemption Amount
    min_redemption: float | None = None
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
        s1 = np.fmax(
            1 + self.bonus_coupon,
            worst_pnl[worst_pnl >= self.coupon_barrier],
        )
        # 最差标的介于 coupon_barrier 和 put_strike 之间
        s2 = np.ones(
            np.count_nonzero(
                (worst_pnl >= self.put_strike) & (worst_pnl < self.coupon_barrier)
            )
        )
        # 最差标的低于 put_strike
        s3 = np.fmax(
            self.min_redemption or 0,
            worst_pnl[worst_pnl < self.put_strike] / self.put_strike,
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
            coupon_barrier=self.coupon_barrier,
            bonus_coupon=self.bonus_coupon,
            min_redemption=self.min_redemption,
        ).price()
        delta = None
        if not price_only:
            epsilon = 0.0005
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
                coupon_barrier=self.coupon_barrier,
                bonus_coupon=self.bonus_coupon,
                min_redemption=self.min_redemption,
            ).price()
            lo_p = type(self)(
                codes=self.codes,
                real_S0=self.real_S0,
                all_relative_S_data=lower,
                T=left_t,
                put_strike=self.put_strike,
                coupon_barrier=self.coupon_barrier,
                bonus_coupon=self.bonus_coupon,
                min_redemption=self.min_redemption,
            ).price()
            delta = (up_p - lo_p) / (2 * epsilon)
        return PriceDelta(price, delta)

    @property
    def discount(self):
        """折现因子"""
        return 1 / (1 + self.r) ** (self.T / 365)
