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
    data_path: None | dict[str, Path] = field(default=None, kw_only=True)
    # 保存全部数据
    all_S_data: None | dict[str, NDArray[Shape["*, *"], Float64]] = field(
        default=None, kw_only=True
    )
    # 模拟路径条数(约定每个标的模拟路径条数相等)
    number_of_paths: int
    # 投资期(以自然日计)
    T: int
    # 这个 T 是自然日数而不是交易日数
    # 但是股价路径是按交易日模拟的
    # 暂时假设用 TD = T * 250 // 365 代表对应的交易日数
    TD: int = field(init=False)
    # key: 标的代码, value: 标的价格路径迭代器函数
    S: dict[str, PriceGenFunc] | dict[str, NDArray[Shape["A, B"], Float64]] = field(
        init=False
    )
    # key: 标的代码, value: 标的初始价格
    S0: dict[str, float] = field(init=False)
    # 标的代码列表
    codes: list[str] = field(init=False)

    def __post_init__(self):
        # 各个标的价格路径迭代器函数

        if self.data_path is None and self.all_S_data is not None:
            self.S = self.all_S_data

            # 各个标的初始价格
            self.S0 = {}
            for k, arr in self.S.items():
                self.S0[k] = arr[0][0]

        elif self.data_path is not None and self.all_S_data is None:
            self.S = {}
            for k, path in self.data_path.items():
                self.S[k] = get_price_path_generator_func(path)

            # 各个标的初始价格
            self.S0 = {}
            for k, gen in self.S.items():
                self.S0[k] = next(gen())[0]

        else:
            raise Exception("输入数据格式错误")

        # 可遍历的 dict_items
        self._Sitems = self.S.items()

        # 标的资产代码列表
        self.codes = list(map(lambda it: it[0], self._Sitems))

    def get_zip_one_path_iterator(
        self,
    ) -> Iterator[tuple[int, NDArray[Shape["A, B"], Float64]]]:
        """将不同底层标的迭代器合并, 返回产出路径编号和路径数组的迭代器"""

        assert self.data_path is not None and self.all_S_data is None, "全部读入数据时, 不可使用迭代"
        # 所有标的未使用的迭代器
        paths_iters = map(lambda it: it[1](), self._Sitems)  # type:ignore
        return map(
            lambda item: (item[0], np.array(item[1])[:, : self.TD + 1]),
            # item[0] 是编号
            # item[1] 是 n 元组, n 是标的数量, 每个元素是一条价格路径
            # 转换成 array 后, 每行是一个标的的一条路径
            # 随后按交易日提取数据
            enumerate(islice(zip(*paths_iters), self.number_of_paths)),
        )

    def get_all_path_arr(self) -> NDArray[Shape["X, Y, Z"], Float64]:
        """获得所有标的的所有路径数据

        Returns
        -------
        NDArray[Shape["X, Y, Z"], Float64]\n
            X = len(self.codes)\n
            Y = self.number_of_paths\n
            Z = self.TD + 1\n
        """

        assert self.data_path is None and self.all_S_data is not None
        price_arr = np.stack(
            # item[1] 二维数据, 0 维是路径数, 1 维是模拟节点数
            list(map(lambda item: item[1][:, : self.TD + 1], self._Sitems))  # type:ignore
        )
        return price_arr
