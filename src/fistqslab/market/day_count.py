"""日期/时间换算约定。

全库统一约定（与重构前版本的口径修正一致）：

- 所有时间参数以 **year fraction（年）** 表示，自然日按 **Act/365** 折算；
- 所有利率为**年化连续复利**，折现因子恒为 ``exp(-r * year_fraction)``；
- 所有波动率为**年化**波动率。

不再使用 ``1 / (1 + r) ** (T/365)`` 或 ``T * 250 // 365`` 这类口径混用。
"""

from __future__ import annotations

_DAYS_PER_YEAR = 365.0


def year_fraction(days: float | int) -> float:
    """把自然日天数换算为年（Act/365）。

    Args:
        days: 自然日天数。

    Returns:
        对应的年数。
    """
    return float(days) / _DAYS_PER_YEAR


def days_from_year_fraction(years: float) -> float:
    """把年换算为自然日天数（Act/365）。

    Args:
        years: 年数。

    Returns:
        对应的自然日天数。
    """
    return float(years) * _DAYS_PER_YEAR
