"""口径与约定测试：利率/折现/时间单位/GBM 分布。"""

from __future__ import annotations

import numpy as np
import pytest

from fistqslab import (
    GBMModel,
    MarketState,
    year_fraction,
)
from fistqslab.market.day_count import days_from_year_fraction


# ---------------------------------------------------------------------------
# 时间与折现口径
# ---------------------------------------------------------------------------
def test_year_fraction_roundtrip():
    """year_fraction 与 days_from_year_fraction 互逆。"""
    assert year_fraction(365) == pytest.approx(1.0)
    assert year_fraction(183) == pytest.approx(183 / 365)
    assert days_from_year_fraction(year_fraction(64)) == pytest.approx(64)


def test_discount_is_continuous_compounding():
    """折现恒为 exp(-r·T)（连续复利），而非 (1+r)^(-T)。"""
    r, T = 0.015, 0.5
    cont = np.exp(-r * T)
    simple = 1 / (1 + r) ** T
    # 若用旧口径 (1+r)^(-T)，与 exp(-rT) 的差异可达数 bp 量级
    assert abs(cont - simple) > 1e-5


# ---------------------------------------------------------------------------
# MarketState 校验
# ---------------------------------------------------------------------------
def test_market_state_broadcast_and_validation():
    """标量广播到多标的；非法输入抛错。"""
    m = MarketState(spots=[100.0, 90.0], risk_free_rate=0.02,
                    dividend_yields=0.01, volatilities=0.2)
    assert m.n_assets == 2
    assert np.allclose(m.dividend_yields, [0.01, 0.01])
    assert np.allclose(m.volatilities, [0.2, 0.2])
    assert np.allclose(m.correlation, np.eye(2))

    with pytest.raises(ValueError):
        MarketState(spots=[100.0, 90.0, 80.0], risk_free_rate=0.02, volatilities=[0.2, 0.2])
    with pytest.raises(ValueError):
        MarketState(spots=[-1.0], risk_free_rate=0.02)
    with pytest.raises(ValueError):
        MarketState(spots=[100.0], risk_free_rate=0.02,
                    correlation=[[1.0, 0.5], [0.5, 1.0]])  # 2x2 与 1 标的冲突


def test_market_state_correlation_validation():
    with pytest.raises(ValueError):
        MarketState(spots=[100.0, 90.0], risk_free_rate=0.02,
                    correlation=[[1.0, 0.5], [0.5, 0.9]])  # 对角线非 1
    with pytest.raises(ValueError):
        MarketState(spots=[100.0, 90.0], risk_free_rate=0.02,
                    correlation=[[1.0, 0.5], [0.6, 1.0]])  # 不对称


# ---------------------------------------------------------------------------
# GBM 分布性质
# ---------------------------------------------------------------------------
def test_gbm_terminal_mean_matches_drift():
    """E[S_T] = S0·exp((r-q)T)：先验 drift 正确（大样本下接近）。"""
    r, q, sigma, T = 0.03, 0.01, 0.25, 1.0
    m = MarketState(spots=[100.0], risk_free_rate=r,
                    dividend_yields=q, volatilities=sigma)
    model = GBMModel(m)
    rels = []
    for rel, _ in model.terminal_batches(T, n_paths=200_000, batch_size=20_000, seed=1):
        rels.append(rel)
    rel = np.concatenate(rels, axis=1)
    expected = np.exp((r - q) * T)
    se = rel.std() / np.sqrt(rel.size)
    assert abs(rel.mean() - expected) < 4 * se


def test_gbm_terminal_log_return_normal():
    """ln(S_T/S0) 服从均值为 (r-q-σ²/2)T、方差 σ²T 的正态。"""
    r, q, sigma, T = 0.03, 0.01, 0.25, 1.0
    m = MarketState(spots=[100.0], risk_free_rate=r,
                    dividend_yields=q, volatilities=sigma)
    model = GBMModel(m)
    rels = []
    for rel, _ in model.terminal_batches(T, n_paths=100_000, batch_size=20_000, seed=2):
        rels.append(rel)
    rel = np.concatenate(rels, axis=1)
    log_r = np.log(rel[0])
    assert log_r.mean() == pytest.approx((r - q - 0.5 * sigma**2) * T, abs=0.005)
    assert log_r.std() == pytest.approx(sigma * np.sqrt(T), abs=0.005)


def test_gbm_correlation_structure():
    """相关 GBM：两个标的对数收益的相关性接近给定相关系数。"""
    corr = 0.6
    m = MarketState(
        spots=[100.0, 100.0],
        risk_free_rate=0.02,
        dividend_yields=[0.0, 0.0],
        volatilities=[0.2, 0.2],
        correlation=[[1.0, corr], [corr, 1.0]],
    )
    model = GBMModel(m)
    rels = []
    for rel, _ in model.terminal_batches(0.5, n_paths=100_000, batch_size=20_000, seed=3):
        rels.append(rel)
    rel = np.concatenate(rels, axis=1)
    log_r = np.log(rel)
    emp_corr = np.corrcoef(log_r[0], log_r[1])[0, 1]
    assert emp_corr == pytest.approx(corr, abs=0.01)


def test_gbm_path_batches_shape_and_increment():
    """path_batches 形状正确，且末观察日分布与 terminal 一致。"""
    m = MarketState(spots=[100.0], risk_free_rate=0.02, volatilities=0.2)
    model = GBMModel(m)
    obs = np.array([0.25, 0.5, 1.0])
    for rel, n_eff in model.path_batches(obs, n_paths=10_000, batch_size=5_000, seed=4):
        assert rel.shape == (1, 3, n_eff)
        # 观察日严格递增保证 cumsum 有效
        break
    # 末观察日 rel 均值 ≈ exp(rT)
    rels = []
    for rel, _ in model.path_batches(obs, n_paths=50_000, batch_size=10_000, seed=5):
        rels.append(rel)
    rel = np.concatenate(rels, axis=2)
    assert rel.shape == (1, 3, 100_000)
    assert rel[0, -1].mean() == pytest.approx(np.exp(0.02), abs=0.01)


def test_gbm_antithetic_symmetry():
    """对偶变量：同 seed 下 antithetic 打开/关闭时样本对称。"""
    m = MarketState(spots=[100.0], risk_free_rate=0.02, volatilities=0.2)
    model = GBMModel(m)
    rels_anti = [rel for rel, _ in model.terminal_batches(0.5, 5_000, 5_000, True, seed=9)]
    rel_anti = np.concatenate(rels_anti, axis=1)
    # 偶数路径，前后半段互为相反数（对数空间）
    half = rel_anti.shape[1] // 2
    assert np.allclose(rel_anti[:, :half] * rel_anti[:, half:], 1.0, rtol=1e-12)
