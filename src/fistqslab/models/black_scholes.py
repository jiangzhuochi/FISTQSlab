"""Black-Scholes 解析定价（含希腊字母）。

统一口径（与全库一致）：

- 所有利率为**年化连续复利**；
- ``T`` 为 **year fraction**（年）；
- 成本持有率 ``b = r - q``（``q`` 为连续股息率）；
- 折现因子恒为 ``exp(-r * T)``。

所有函数支持向量化：``S`` 可为标量或任意形状数组。
``T`` 应为严格正的小量（如 ``1e-8``）而非 0，否则 ``d1/d2`` 趋于无穷。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm as _norm

__all__ = [
    "bs_call",
    "bs_put",
    "bs_digital_call",
    "bs_digital_put",
    "bs_greeks",
    "bs_call_from_greeks",
]


def _prepare(
    S: np.ndarray | float,
    K: float,
    T: float,
    r: float,
    sigma: np.ndarray | float,
    q: float,
) -> tuple[Any, Any, Any, Any]:
    """计算 d1/d2 与两个折现因子。"""
    b = r - q
    sqrt_t = np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    # e^{(b-r)T}：含股息的远期折现（用于 S 项）
    discount = np.exp((b - r) * T)
    # e^{-rT}：纯利率折现（用于 K 项）
    discount_t = np.exp(-r * T)
    return d1, d2, discount, discount_t


def bs_call(
    S: np.ndarray | float,
    K: float,
    T: float,
    r: float,
    sigma: np.ndarray | float,
    q: float = 0.0,
) -> np.ndarray | float:
    """欧式认购期权价格。"""
    d1, d2, discount, discount_t = _prepare(S, K, T, r, sigma, q)
    return S * discount * _norm.cdf(d1) - K * discount_t * _norm.cdf(d2)


def bs_put(
    S: np.ndarray | float,
    K: float,
    T: float,
    r: float,
    sigma: np.ndarray | float,
    q: float = 0.0,
) -> np.ndarray | float:
    """欧式认沽期权价格。"""
    d1, d2, discount, discount_t = _prepare(S, K, T, r, sigma, q)
    return K * discount_t * _norm.cdf(-d2) - S * discount * _norm.cdf(-d1)


def bs_digital_call(
    S: np.ndarray | float,
    K: float,
    T: float,
    r: float,
    sigma: np.ndarray | float,
    q: float = 0.0,
    cash: float = 1.0,
) -> np.ndarray | float:
    """现金或无认购期权（payoff = cash * 1{S_T > K}）。"""
    _, d2, _, discount_t = _prepare(S, K, T, r, sigma, q)
    return cash * discount_t * _norm.cdf(d2)


def bs_digital_put(
    S: np.ndarray | float,
    K: float,
    T: float,
    r: float,
    sigma: np.ndarray | float,
    q: float = 0.0,
    cash: float = 1.0,
) -> np.ndarray | float:
    """现金或无认沽期权（payoff = cash * 1{S_T < K}）。"""
    _, d2, _, discount_t = _prepare(S, K, T, r, sigma, q)
    return cash * discount_t * _norm.cdf(-d2)


def bs_greeks(
    option: str,
    S: np.ndarray | float,
    K: float,
    T: float,
    r: float,
    sigma: np.ndarray | float,
    q: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """欧式期权的价格与希腊字母。

    Args:
        option: ``"call"`` / ``"put"`` / ``"digital_call"`` / ``"digital_put"``。
        S, K, T, r, sigma, q: 同 ``bs_call``。

    Returns:
        dict，包含 ``price``/``delta``/``gamma``/``vega``/``theta``/``rho``。
        theta 为时间流逝方向（``∂V/∂t``），vanilla 一般为负。
    """
    d1, d2, discount, discount_t = _prepare(S, K, T, r, sigma, q)
    sqrt_t = np.sqrt(T)
    n = _norm.pdf
    N = _norm.cdf

    if option == "call":
        price = bs_call(S, K, T, r, sigma, q)
        delta = discount * N(d1)
        gamma = discount * n(d1) / (S * sigma * sqrt_t)
        vega = discount * S * sqrt_t * n(d1)
        # b = r - q，故 (b - r) = -q：
        # theta = -e^{(b-r)T}·S·σ·n(d1)/(2√T) - (b-r)·S·e^{(b-r)T}·N(d1) - K·r·e^{-rT}·N(d2)
        theta = (
            -discount * S * sigma * n(d1) / (2 * sqrt_t)
            + q * S * discount * N(d1)
            - K * r * discount_t * N(d2)
        )
        rho = T * K * discount_t * N(d2)
    elif option == "put":
        price = bs_put(S, K, T, r, sigma, q)
        delta = discount * (N(d1) - 1)
        gamma = discount * n(d1) / (S * sigma * sqrt_t)
        vega = discount * S * sqrt_t * n(d1)
        theta = (
            -discount * S * sigma * n(d1) / (2 * sqrt_t)
            - q * S * discount * N(-d1)
            + K * r * discount_t * N(-d2)
        )
        rho = -T * K * discount_t * N(-d2)
    elif option == "digital_call":
        price = bs_digital_call(S, K, T, r, sigma, q)
        factor = discount_t * n(d2)
        delta = factor / (S * sigma * sqrt_t)
        gamma = -factor * d1 / (S**2 * sigma**2 * T)
        vega = -factor * d1 / sigma
        theta = factor * (d1 / (2 * T) - (r - q) / (sigma * sqrt_t)) + r * price
        # ∂/∂r [cash·e^{-rT}·N(d2)] = -T·price + cash·e^{-rT}·n(d2)·√T/σ
        rho = -T * price + factor * sqrt_t / sigma
    elif option == "digital_put":
        price = bs_digital_put(S, K, T, r, sigma, q)
        factor = discount_t * n(d2)
        delta = -factor / (S * sigma * sqrt_t)
        gamma = factor * d1 / (S**2 * sigma**2 * T)
        vega = factor * d1 / sigma
        theta = -factor * (d1 / (2 * T) - (r - q) / (sigma * sqrt_t)) + r * price
        rho = -T * price - factor * sqrt_t / sigma
    else:
        raise ValueError(f"未知期权类型: {option!r}")

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def bs_call_from_greeks(
    S: np.ndarray | float,
    K: float,
    T: float,
    r: float,
    sigma: np.ndarray | float,
    q: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """便捷函数：认购期权的价格与希腊字母。"""
    return bs_greeks("call", S, K, T, r, sigma, q)
