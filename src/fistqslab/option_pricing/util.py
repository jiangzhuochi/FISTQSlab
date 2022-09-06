import json
from functools import partial

import numpy as np
import numpy.typing as npt
import pandas as pd


class DataFrameJSONEncoder(json.JSONEncoder):
    def default(self, o: pd.DataFrame):
        try:
            return o.to_dict()
        except TypeError:
            pass
        return json.JSONEncoder.default(self, o)


def get_stock_path(
    S0: float, mu: float, vol: float, u: int, m_step: int, n_path: int = 100000
) -> npt.NDArray[np.float64]:
    """BS 公式生成股价路径。

    Parameters
    ----------
    S0 : float
        期初价格
    mu : float
        每单位时间收益率期望
    vol : float
        每单位时间收益率标准差
    u : int
        共有多少个单位时间
    m_step : int
        每单位时间模拟的步数
    n_path : int, optional
        模拟路径的条数, by default 100000

    Returns
    -------
    npt.NDArray[np.float64]
        形状为 ((m_step*u)+1), 2*n_path 的 numpy 数组
        2*n_path 列，每列是一条路径

    步骤解释:
    产生随机数   映射在指数上   第一行赋值为1       累乘后乘以期初价
    e0       ->  exp(f(e0))   ->  1            ->  S0 * 1
    e1       ->  exp(f(e1))   ->  exp(f(e1))   ->  S0 * exp(f(e1))
    e2       ->  exp(f(e2))   ->  exp(f(e2))   ->  S0 * exp(f(e1))*exp(f(e2))
    .        .                .                .
    .        .                .                .
    emu      ->  exp(f(emu))  ->  exp(f(emu))  ->  S0 * exp(f(e1))*exp(f(e2))*...*exp(f(emu))

    其中 f(e) = np.exp((mu - 0.5 * vol**2) * delta_t + vol * np.sqrt(delta_t) * e)
    """

    delta_t = 1 / m_step
    # 产生标准正态分布随机数
    _half_ep = np.random.normal(size=((m_step * u) + 1, n_path))
    # 对偶变量法, 加上随机数的相反数
    epsilons = np.hstack([_half_ep, -_half_ep])
    # 直接用运算符, df 的 applymap 时间要几百倍
    multiplier = np.exp(
        (mu - 0.5 * vol**2) * delta_t + vol * np.sqrt(delta_t) * epsilons
    )
    # 第一行赋值为 1, 用作期初价格的乘子
    multiplier[0] = 1
    # 累乘后乘以期初价
    stock_path = np.cumprod(multiplier, axis=0) * S0

    return stock_path


# 1 年股价路径
get_stock_path_1y = partial(get_stock_path, u=1, m_step=250)
