from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Iterator

import numpy as np
from nptyping import Float64, NDArray, Shape

from .util import PriceGenFunc, get_price_path_generator_func


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

    def get_zip_one_path_iterator(
        self,
    ) -> Iterator[tuple[int, NDArray[Shape["A, B"], Float64]]]:
        """将不同底层标的迭代器合并, 返回产出路径编号和路径数组的迭代器"""

        # 所有标的未使用的迭代器
        paths_iters = map(lambda it: it[1](), self._Sitems)
        return map(
            lambda item: (item[0], np.array(item[1])),
            # item[0] 是编号
            # item[1] 是 n 元组, n 是标的数量, 每个元素是一条价格路径
            # 转换成 array 后, 每行是一个标的的一条路径
            enumerate(islice(zip(*paths_iters), self.number_of_paths)),
        )
