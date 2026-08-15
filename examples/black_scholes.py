"""Black-Scholes 解析定价示例：价格与希腊字母。

运行：python examples/black_scholes.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from fistqslab import bs_call, bs_greeks, bs_put


def main() -> None:
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    print("=" * 56)
    print(f"S={S}  K={K}  T={T}y  r={r}  sigma={sigma}")
    print("=" * 56)
    c = bs_call(S, K, T, r, sigma)
    p = bs_put(S, K, T, r, sigma)
    print(f"Call = {c:10.6f}   Put = {p:10.6f}")
    print(f"Put-Call parity: C-P = {c - p:10.6f}  vs S - K·e^-rT = {S - K * np.exp(-r * T):10.6f}")

    g = bs_greeks("call", S, K, T, r, sigma)
    print("\nCall 希腊字母（解析）：")
    for name, val in g.items():
        print(f"  {name:>6s} = {val:.6f}")

    # 有限差分校验 delta/gamma
    h = 1e-4
    delta_fd = (bs_call(S + h, K, T, r, sigma) - bs_call(S - h, K, T, r, sigma)) / (2 * h)
    gamma_fd = (bs_call(S + h, K, T, r, sigma) - 2 * c + bs_call(S - h, K, T, r, sigma)) / h**2
    print(f"\n有限差分校验：delta_fd = {delta_fd:.6f}，gamma_fd = {gamma_fd:.6f}")

    # 概率：P(S_T > K)
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    print(f"\nP(S_T > K) = {norm.cdf(d2):.4f}")


if __name__ == "__main__":
    main()
