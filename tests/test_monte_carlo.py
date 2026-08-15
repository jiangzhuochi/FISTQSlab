"""Monte Carlo 引擎测试：MC vs BS、分块一致性、可复现性、收敛、内存。"""

from __future__ import annotations

import numpy as np
import pytest

from fistqslab import (
    AnalyticEngine,
    GBMModel,
    MarketState,
    MonteCarloEngine,
)
from fistqslab.models.black_scholes import bs_call
from fistqslab.products.vanilla import EuropeanCallOption, EuropeanPutOption


@pytest.fixture(scope="module")
def market():
    return MarketState(spots=[100.0], risk_free_rate=0.02, volatilities=0.25)


@pytest.fixture(scope="module")
def call_product():
    return EuropeanCallOption(strike=100.0, maturity_year_fraction=1.0)


def test_mc_matches_bs_within_3sigma(market, call_product):
    """MC 价格与 BS 解析值之差 < 3 × stderr。"""
    model = GBMModel(market)
    engine = MonteCarloEngine(n_paths=100_000, batch_size=10_000, seed=11)
    result = engine.price(call_product, model)
    bs = bs_call(100.0, 100.0, 1.0, 0.02, 0.25)
    assert abs(result.price - bs) < 3 * result.stderr


def test_mc_put_call_consistency(market):
    """MC 下 call 与 put 满足 parity：C - P ≈ S - K·e^{-rT}。"""
    model = GBMModel(market)
    engine = MonteCarloEngine(n_paths=80_000, batch_size=10_000, seed=12)
    c = engine.price(EuropeanCallOption(95.0, 0.5), model).price
    p = engine.price(EuropeanPutOption(95.0, 0.5), model).price
    parity = 100.0 - 95.0 * np.exp(-0.02 * 0.5)
    assert abs((c - p) - parity) < 3 * 0.02


def test_chunked_equals_single_batch(market, call_product):
    """同 seed 下分块与单批结果逐位一致（引擎只做累加）。"""
    model = GBMModel(market)
    chunked = MonteCarloEngine(n_paths=30_000, batch_size=3_000, seed=21).price(call_product, model)
    single = MonteCarloEngine(n_paths=30_000, batch_size=30_000, seed=21).price(call_product, model)
    assert chunked.price == single.price
    assert chunked.n_effective_paths == single.n_effective_paths


def test_seed_reproducibility(market, call_product):
    """相同 seed 结果逐位一致；不同 seed 结果接近。"""
    model = GBMModel(market)
    r1 = MonteCarloEngine(n_paths=20_000, seed=31).price(call_product, model)
    r2 = MonteCarloEngine(n_paths=20_000, seed=31).price(call_product, model)
    r3 = MonteCarloEngine(n_paths=20_000, seed=32).price(call_product, model)
    assert r1.price == r2.price
    assert abs(r1.price - r3.price) < 4 * r1.stderr


def test_more_paths_reduces_stderr(market, call_product):
    """路径数越多 stderr 越小。"""
    model = GBMModel(market)
    small = MonteCarloEngine(n_paths=5_000, seed=41).price(call_product, model)
    large = MonteCarloEngine(n_paths=80_000, seed=42).price(call_product, model)
    assert large.stderr < small.stderr / 2


def test_antithetic_symmetry_engine(market, call_product):
    """对偶变量打开/关闭时价格一致（同 seed 下样本分布相同）。"""
    model = GBMModel(market)
    on = MonteCarloEngine(n_paths=20_000, antithetic=True, seed=51).price(call_product, model)
    off = MonteCarloEngine(n_paths=20_000, antithetic=False, seed=51).price(call_product, model)
    # 同 seed 下 on 的前半段 Z 与 off 相同；on 还包含 -Z 对称路径，均值估计略有差异
    assert on.price == pytest.approx(off.price, abs=0.06)
    assert on.n_effective_paths == 40_000
    assert off.n_effective_paths == 20_000


def test_million_paths_smoke(market, call_product):
    """100 万路径 + 小 batch：内存恒定且能完成（冒烟）。"""
    model = GBMModel(market)
    result = MonteCarloEngine(
        n_paths=1_000_000, batch_size=20_000, seed=61
    ).price(call_product, model)
    bs = bs_call(100.0, 100.0, 1.0, 0.02, 0.25)
    assert abs(result.price - bs) < 3 * result.stderr
    assert result.n_effective_paths == 2_000_000


def test_result_metadata(market, call_product):
    """PricingResult 携带 stderr / CI / seed / 路径数。"""
    model = GBMModel(market)
    r = MonteCarloEngine(n_paths=10_000, seed=71).price(call_product, model)
    assert r.method == "monte_carlo"
    assert r.seed == 71
    assert r.n_paths == 10_000
    assert r.n_effective_paths == 20_000
    assert r.ci_low < r.price < r.ci_high
    assert r.stderr > 0


def test_analytic_engine(market, call_product):
    """解析引擎返回无 stderr 的结果；无闭式解时抛错。"""
    ae = AnalyticEngine()
    r = ae.price(call_product, market)
    assert r.method == "analytic"
    assert r.stderr is None
    assert r.price == pytest.approx(bs_call(100.0, 100.0, 1.0, 0.02, 0.25), abs=1e-9)

    class NoAnalytic(EuropeanCallOption):
        def analytic_price(self, market):
            return None

    with pytest.raises(NotImplementedError):
        ae.price(NoAnalytic(100.0, 1.0), market)


def test_invalid_engine_params():
    with pytest.raises(ValueError):
        MonteCarloEngine(n_paths=0)
    with pytest.raises(ValueError):
        MonteCarloEngine(n_paths=100, batch_size=0)
