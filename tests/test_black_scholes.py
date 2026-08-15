"""Black-Scholes 解析定价基准测试。

覆盖：已知基准值、put-call parity、希腊字母 vs 有限差分、
边界行为（T→0 / σ→0 / deep ITM / OTM）、单调性质、digital 公式。
"""

from __future__ import annotations

import numpy as np
import pytest

from fistqslab.models.black_scholes import (
    bs_call,
    bs_digital_call,
    bs_digital_put,
    bs_greeks,
    bs_put,
)


# ---------------------------------------------------------------------------
# 基准值（S=100, K=100, T=1, r=0.05, σ=0.2, q=0）
# ---------------------------------------------------------------------------
def test_call_put_reference_values():
    """已知基准：ATM call ≈ 10.4506, put ≈ 5.5735（q=0）。"""
    c = bs_call(100.0, 100.0, 1.0, 0.05, 0.2)
    p = bs_put(100.0, 100.0, 1.0, 0.05, 0.2)
    assert c == pytest.approx(10.450583572185565, abs=1e-9)
    assert p == pytest.approx(5.573526022256971, abs=1e-9)


@pytest.mark.parametrize(
    ("S", "K", "T", "r", "sigma", "q"),
    [
        (100.0, 100.0, 1.0, 0.05, 0.2, 0.0),
        (95.0, 105.0, 0.5, 0.03, 0.25, 0.0),
        (120.0, 90.0, 2.0, 0.04, 0.3, 0.02),
        (80.0, 110.0, 0.25, 0.06, 0.15, 0.01),
    ],
)
def test_put_call_parity(S, K, T, r, sigma, q):
    """C - P = S·e^{-qT} - K·e^{-rT}。"""
    c = bs_call(S, K, T, r, sigma, q)
    p = bs_put(S, K, T, r, sigma, q)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-9)


def test_vectorized_s():
    """S 为数组时返回逐元素价格。"""
    S = np.array([80.0, 100.0, 120.0])
    c = np.asarray(bs_call(S, 100.0, 1.0, 0.05, 0.2))
    assert c.shape == (3,)
    for s, val in zip(S, c, strict=True):
        assert val == pytest.approx(bs_call(s, 100.0, 1.0, 0.05, 0.2), abs=1e-12)


# ---------------------------------------------------------------------------
# 希腊字母 vs 有限差分
# ---------------------------------------------------------------------------
def _fd(f, x, h, axis=0):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_greeks_match_finite_difference():
    """解析希腊字母与中心差分一致（delta/gamma/vega/theta/rho）。"""
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.01
    for option in ("call", "put"):
        g = bs_greeks(option, S, K, T, r, sigma, q)
        h = 1e-4
        assert g["delta"] == pytest.approx(
            _fd(
                lambda s: (
                    bs_call(s, K, T, r, sigma, q)
                    if option == "call"
                    else bs_put(s, K, T, r, sigma, q)
                ),
                S,
                h,
            ),
            rel=1e-4,
        )
        assert g["gamma"] == pytest.approx(
            (
                (
                    bs_call(S + h, K, T, r, sigma, q)
                    if option == "call"
                    else bs_put(S + h, K, T, r, sigma, q)
                )
                - 2
                * (
                    bs_call(S, K, T, r, sigma, q)
                    if option == "call"
                    else bs_put(S, K, T, r, sigma, q)
                )
                + (
                    bs_call(S - h, K, T, r, sigma, q)
                    if option == "call"
                    else bs_put(S - h, K, T, r, sigma, q)
                )
            )
            / h**2,
            rel=1e-3,
        )
        assert g["vega"] == pytest.approx(
            _fd(
                lambda v: (
                    bs_call(S, K, T, r, v, q)
                    if option == "call"
                    else bs_put(S, K, T, r, v, q)
                ),
                sigma,
                1e-4,
            ),
            rel=1e-4,
        )
        assert g["theta"] == pytest.approx(
            -_fd(
                lambda t: (
                    bs_call(S, K, t, r, sigma, q)
                    if option == "call"
                    else bs_put(S, K, t, r, sigma, q)
                ),
                T,
                1e-4,
            ),
            rel=1e-4,
        )
        assert g["rho"] == pytest.approx(
            _fd(
                lambda x: (
                    bs_call(S, K, T, x, sigma, q)
                    if option == "call"
                    else bs_put(S, K, T, x, sigma, q)
                ),
                r,
                1e-4,
            ),
            rel=1e-4,
        )


def test_greeks_against_reference():
    """参考希腊值（S=100, K=100, T=1, r=0.05, σ=0.2, q=0）。"""
    g = bs_greeks("call", 100.0, 100.0, 1.0, 0.05, 0.2)
    assert g["delta"] == pytest.approx(0.636831, abs=1e-6)
    assert g["gamma"] == pytest.approx(0.018762, abs=1e-6)
    assert g["vega"] == pytest.approx(37.5240, abs=1e-3)
    assert g["theta"] == pytest.approx(-6.4140, abs=1e-3)
    assert g["rho"] == pytest.approx(53.2323, abs=1e-3)


# ---------------------------------------------------------------------------
# 边界行为
# ---------------------------------------------------------------------------
def test_boundary_t_to_zero():
    """T→0：call → max(S-K, 0)。"""
    T = 1e-10
    assert bs_call(110.0, 100.0, T, 0.05, 0.2) == pytest.approx(10.0, abs=1e-6)
    assert bs_call(90.0, 100.0, T, 0.05, 0.2) == pytest.approx(0.0, abs=1e-6)
    assert bs_put(90.0, 100.0, T, 0.05, 0.2) == pytest.approx(10.0, abs=1e-6)


def test_boundary_sigma_to_zero():
    """σ→0：ITM call → S - K·e^{-rT}。"""
    sigma = 1e-10
    assert bs_call(110.0, 100.0, 1.0, 0.05, sigma) == pytest.approx(
        110.0 - 100.0 * np.exp(-0.05), abs=1e-6
    )


def test_deep_itm_otm_monotonic():
    """性质：S↑ => call 不降；σ↑ => call/put 不降。"""
    S_grid = np.linspace(50, 150, 51)
    c = bs_call(S_grid, 100.0, 1.0, 0.05, 0.2)
    p = bs_put(S_grid, 100.0, 1.0, 0.05, 0.2)
    assert np.all(np.diff(c) >= 0)
    assert np.all(np.diff(p) <= 0)

    sigma_grid = np.linspace(0.01, 0.8, 50)
    c_sig = bs_call(100.0, 100.0, 1.0, 0.05, sigma_grid)
    p_sig = bs_put(100.0, 100.0, 1.0, 0.05, sigma_grid)
    assert np.all(np.diff(c_sig) >= 0)
    assert np.all(np.diff(p_sig) >= 0)


def test_dividend_reduces_call():
    """股息 q↑ => call 下降（股利拖累标的远期）。"""
    c0 = bs_call(100.0, 100.0, 1.0, 0.05, 0.2, q=0.0)
    c1 = bs_call(100.0, 100.0, 1.0, 0.05, 0.2, q=0.05)
    assert c1 < c0


# ---------------------------------------------------------------------------
# Digital
# ---------------------------------------------------------------------------
def test_digital_call_formula():
    """cash-or-nothing call = cash·e^{-rT}·N(d2)；put = cash·e^{-rT}·N(-d2)。"""
    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.03, 0.2
    cash = 2.0
    from scipy.stats import norm

    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    assert bs_digital_call(S, K, T, r, sigma, cash=cash) == pytest.approx(
        cash * np.exp(-r * T) * norm.cdf(d2), abs=1e-12
    )
    assert bs_digital_put(S, K, T, r, sigma, cash=cash) == pytest.approx(
        cash * np.exp(-r * T) * norm.cdf(-d2), abs=1e-12
    )
    # call + put = 折现的现金（二元期权互斥覆盖全部状态）
    assert (
        bs_digital_call(S, K, T, r, sigma, cash=cash)
        + bs_digital_put(S, K, T, r, sigma, cash=cash)
    ) == pytest.approx(cash * np.exp(-r * T), abs=1e-12)


def test_digital_greeks_match_fd():
    """digital 希腊字母与有限差分一致。"""
    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.03, 0.2
    for option in ("digital_call", "digital_put"):
        g = bs_greeks(option, S, K, T, r, sigma)
        fn = lambda s: bs_digital_call(s, K, T, r, sigma)
        fp = lambda s: bs_digital_put(s, K, T, r, sigma)
        f = fn if option == "digital_call" else fp
        h = 1e-4
        assert g["delta"] == pytest.approx(_fd(f, S, h), rel=1e-4)
        assert g["gamma"] == pytest.approx(
            (f(S + h) - 2 * f(S) + f(S - h)) / h**2,
            rel=1e-3,
        )
        assert g["vega"] == pytest.approx(
            _fd(
                lambda v: (
                    bs_digital_call(S, K, T, r, v)
                    if option == "digital_call"
                    else bs_digital_put(S, K, T, r, v)
                ),
                sigma,
                1e-4,
            ),
            rel=1e-4,
        )
        assert g["theta"] == pytest.approx(
            -_fd(
                lambda t: (
                    bs_digital_call(S, K, t, r, sigma)
                    if option == "digital_call"
                    else bs_digital_put(S, K, t, r, sigma)
                ),
                T,
                1e-4,
            ),
            rel=1e-4,
        )
        assert g["rho"] == pytest.approx(
            _fd(
                lambda x: (
                    bs_digital_call(S, K, T, x, sigma)
                    if option == "digital_call"
                    else bs_digital_put(S, K, T, x, sigma)
                ),
                r,
                1e-4,
            ),
            rel=1e-4,
        )


def test_bs_accepts_arrays_and_scalars():
    """标量与数组输入兼容。"""
    s = 100.0
    arr = np.array([100.0, 100.0])
    assert np.allclose(
        bs_call(arr, 100.0, 1.0, 0.05, 0.2), bs_call(s, 100.0, 1.0, 0.05, 0.2)
    )
    g = bs_greeks("call", s, 100.0, 1.0, 0.05, 0.2)
    assert set(g) == {"price", "delta", "gamma", "vega", "theta", "rho"}


def test_unknown_option_raises():
    with pytest.raises(ValueError):
        bs_greeks("straddle", 100.0, 100.0, 1.0, 0.05, 0.2)
