import json
from functools import partial
from pathlib import Path
from typing import Callable, Generator

import numexpr as ne
import numpy as np
import pandas as pd
from nptyping import Float64, Int, NDArray, Shape


class DataFrameJSONEncoder(json.JSONEncoder):
    def default(self, o: pd.DataFrame):
        try:
            return o.to_dict()
        except TypeError:
            pass
        return json.JSONEncoder.default(self, o)


# TODO: 需要转置
def get_stock_path(
    S0: float, mu: float, vol: float, u: int, m_step: int, n_path: int = 100000
) -> NDArray[Shape["*, *"], Float64]:
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
    # 耗时大约减少了 15%
    multiplier = ne.evaluate(
        """exp(
        (mu - 0.5 * vol**2) * delta_t + vol * sqrt(delta_t) * epsilons
    )"""
    )
    # 第一行赋值为 1, 用作期初价格的乘子
    multiplier[0] = 1
    # 累乘后乘以期初价
    stock_path = np.cumprod(multiplier, axis=0) * S0

    return stock_path


# 1 年股价路径(单位时间为天, 共250天, 每天240步)
get_stock_path_1y = partial(get_stock_path, u=250, m_step=240)

PriceGenFunc = Callable[[], Generator[NDArray[Shape["*"], Float64], None, None]]


def get_price_path_generator_func(csv_path: Path) -> PriceGenFunc:
    """返回一个迭代器函数 g, 调用函数 g, 返回价格路径的迭代器

    Parameters
    -------
    csv_path : Path
        csv 路径, 该文件每行存储一条价格路径
    """

    def g() -> Generator[NDArray[Shape["*"], Float64], None, None]:
        with open(csv_path) as f:
            for line in f:
                yield np.array(line.split(","), dtype=float)

    return g


def cmp_dict_all_items(d1: dict, d2: dict, op: Callable):
    """比较两个键相同的字典, 它们的值是否全部符合大小关系

    Parameters
    ----------
    d1 : dict
        左边的字典
    d2 : dict
        右边的字典
    op : Callable
        比较符号, operator 库里的大小函数

    Returns
    -------
    bool
        是否满足给定条件
    """

    assert d1
    assert d1.keys() == d2.keys()

    for key in d1:
        if not op(d1[key], d2[key]):
            return False
    return True
