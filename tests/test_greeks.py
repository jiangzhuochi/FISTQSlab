"""Greeks 测试：有限差分 vs 解析、CRN 复现、产品一致性。"""

from __future__ import annotations

import numpy as np
import pytest

from fistqslab import (
    GBMModel,
    GreeksCalculator,
    LeverageNote,
    MarketState,
    MonteCarloEngine,
)
from fistqslab.models.black_scholes import bs_greeks
from fistqslab.products.vanilla import EuropeanCallOption, EuropeanPutOption

MARKET = MarketState(spots=[100.0], risk_free_rate=0.02, volatilities=0.25)
T = 0.5


def _calculator(n_paths=100_000, seed=101):
    return GreeksCalculator(MonteCarloEngine(n_paths=n_paths, seed=seed))


def test_vanilla_delta_matches_analytic():
    """vanilla call delta（对 S0 的导数）≈ 解析 delta。"""
    call = EuropeanCallOption(strike=100.0, maturity_year_fraction=T)
    model = GBMModel(MARKET)
    g = _calculator().delta(call, model)
    bs = bs_greeks("call", 100.0, 100.0, T, 0.02, 0.25)
    assert g[0] == pytest.approx(bs["delta"], abs=0.01)


def test_vanilla_gamma_matches_analytic():
    """vanilla call gamma ≈ 解析 gamma。"""
    call = EuropeanCallOption(strike=100.0, maturity_year_fraction=T)
    model = GBMModel(MARKET)
    g = _calculator().gamma(call, model)
    bs = bs_greeks("call", 100.0, 100.0, T, 0.02, 0.25)
    assert g[0] == pytest.approx(bs["gamma"], abs=0.002)


def test_put_delta_matches_analytic():
    """vanilla put delta ≈ 解析 delta（负值）。"""
    put = EuropeanPutOption(strike=100.0, maturity_year_fraction=T)
    model = GBMModel(MARKET)
    g = _calculator().delta(put, model)
    bs = bs_greeks("put", 100.0, 100.0, T, 0.02, 0.25)
    assert g[0] == pytest.approx(bs["delta"], abs=0.01)


def test_crn_reproducibility():
    """CRN：相同 seed 下 Greeks 逐位一致（差分路径同源）。"""
    call = EuropeanCallOption(strike=100.0, maturity_year_fraction=T)
    model = GBMModel(MARKET)
    d1 = _calculator(seed=111).delta(call, model)
    d2 = _calculator(seed=111).delta(call, model)
    d3 = _calculator(seed=222).delta(call, model)
    assert np.allclose(d1, d2)
    # 不同 seed 结果应接近（都在解析值附近）
    assert np.allclose(d1, d3, atol=0.02)


def test_leverage_note_delta_is_zero():
    """杠杆票据 payoff 只依赖相对收益，delta 对绝对 spot ≈ 0。"""
    lev = LeverageNote(leverage_multiple=2.0, maturity_year_fraction=T,
                       dividend_rate=0.04)
    model = GBMModel(MARKET)
    d = _calculator().delta(lev, model)
    assert abs(d[0]) < 0.005


def test_multi_asset_delta_shape():
    """多标的产品的 delta 为逐标的向量。"""
    from fistqslab import ELN

    m2 = MarketState(spots=[100.0, 90.0], risk_free_rate=0.02,
                     volatilities=[0.25, 0.3],
                     correlation=[[1.0, 0.5], [0.5, 1.0]])
    eln = ELN(strike=0.95, maturity_year_fraction=T)
    model = GBMModel(m2)
    d, gam = _calculator(n_paths=60_000).delta_gamma(eln, model)
    assert d.shape == (2,)
    assert gam.shape == (2,)
    assert np.isfinite(d).all() and np.isfinite(gam).all()
