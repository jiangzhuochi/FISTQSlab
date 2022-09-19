from dataclasses import dataclass, field

import numpy as np
from nptyping import Float64, NDArray, Shape

from .abc_option import BaseOption
from .mc import MonteCarlo
from .util import find_worst_target


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
