"""结构化产品测试：确定路径手算 payoff、MC vs 闭式、未用参数回归。"""

from __future__ import annotations

import numpy as np
import pytest

from fistqslab import (
    ELN,
    RELN,
    AutoCallNote,
    BonusEnhancedNote,
    GBMModel,
    LeverageNote,
    MarketState,
    MonteCarloEngine,
    RainbowNote,
    get_eln_strike_from_issue_price,
    get_reln_issue_price,
)

R = 0.02
MARKET = MarketState(spots=[100.0], risk_free_rate=R, volatilities=0.25)
MARKET2 = MarketState(
    spots=[100.0, 90.0],
    risk_free_rate=R,
    volatilities=[0.25, 0.3],
    correlation=[[1.0, 0.5], [0.5, 1.0]],
)


def _eln_scenarios():
    eln = ELN(strike=0.95, maturity_year_fraction=0.5)
    cases = [
        (np.array([[1.1]]), 1.0),          # 大涨：全额本金
        (np.array([[0.95]]), 1.0),         # 平值：不转股
        (np.array([[0.8]]), 1 - 0.15 / 0.95),  # 转股
        (np.array([[1.0], [0.9]]), 1 - 0.05 / 0.95),  # worst-of 多标的
    ]
    for rel, expected in cases:
        assert eln.payoff_terminal(rel, np.ones(rel.shape[0])) == pytest.approx(
            expected, abs=1e-9
        )


def test_eln_payoff_handcrafted():
    _eln_scenarios()


def _reln_scenarios():
    reln = RELN(strike=1.035, issue_price=1.0172, maturity_year_fraction=0.5)
    cases = [
        (np.array([[1.2]]), 1.035 + 1.0172 - 1.0),
        (np.array([[1.035]]), 1.035 + 1.0172 - 1.0),
        (np.array([[0.9]]), 1.035 + 1.0172 - 1.0 - 0.135),
    ]
    for rel, expected in cases:
        assert reln.payoff_terminal(rel, np.ones(rel.shape[0])) == pytest.approx(
            expected, abs=1e-9
        )


def test_reln_payoff_handcrafted():
    _reln_scenarios()


def _ben_scenarios():
    ben = BonusEnhancedNote(
        put_strike=0.9, coupon_barrier=1.0, bonus_coupon=0.05,
        maturity_year_fraction=0.5,
    )
    cases = [
        (np.array([[1.2]]), 1.0 + 0.15 + 0.05),        # 大涨
        (np.array([[0.95]]), 1.0),                      # 未达票息门槛
        (np.array([[0.8]]), 0.8 / 0.9),                 # 转股
        (np.array([[0.9]]), 1.0),                       # 平值
    ]
    for rel, expected in cases:
        assert ben.payoff_terminal(rel, np.ones(rel.shape[0])) == pytest.approx(
            expected, abs=1e-9
        )

    ben_mr = BonusEnhancedNote(
        put_strike=0.9, coupon_barrier=1.0, bonus_coupon=0.05,
        maturity_year_fraction=0.5, min_redemption=0.8,
    )
    cases_mr = [
        (np.array([[1.2]]), 1.0 + 0.15 + 0.05),
        (np.array([[0.85]]), 0.8 + 0.13 / 0.9),         # 保本平台内
        (np.array([[0.72]]), 0.8),                      # 保本下限
        (np.array([[0.6]]), 0.8),                       # 深跌破保本 → 仍保本 0.8
    ]
    for rel, expected in cases_mr:
        assert ben_mr.payoff_terminal(rel, np.ones(rel.shape[0])) == pytest.approx(
            expected, abs=1e-9
        )


def test_ben_payoff_handcrafted():
    _ben_scenarios()


def test_rainbow_payoff_handcrafted():
    rb = RainbowNote(
        put_strike=0.9, lower_call_strike=1.0, upside_participation=0.8,
        guaranteed_flat_coupon=0.04, maturity_year_fraction=0.5,
    )
    cases = [
        (np.array([[1.3]]), 1.04 + 0.8 * 0.3),          # 参与率上行
        (np.array([[0.95]]), 1.04),                      # 平台
        (np.array([[0.8]]), 1.04 - 0.1 / 0.9),           # 转股
        (np.array([[1.0]]), 1.04),
    ]
    for rel, expected in cases:
        assert rb.payoff_terminal(rel, np.ones(rel.shape[0])) == pytest.approx(
            expected, abs=1e-9
        )


def test_leverage_payoff_handcrafted():
    lev = LeverageNote(
        leverage_multiple=2.0, maturity_year_fraction=0.5, dividend_rate=0.04
    )
    cases = [
        (np.array([[1.2]]), 1.0 + 2.0 * (0.2 + 0.02)),
        (np.array([[1.0]]), 1.0 + 2.0 * 0.02),
        (np.array([[0.85]]), 1.0 + 2.0 * (-0.15 + 0.02)),
    ]
    for rel, expected in cases:
        assert lev.payoff_terminal(rel, np.ones(rel.shape[0])) == pytest.approx(
            expected, abs=1e-9
        )


# ---------------------------------------------------------------------------
# MC vs 闭式
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "product",
    [
        ELN(strike=0.95, maturity_year_fraction=0.5),
        RELN(strike=1.035, issue_price=1.0172, maturity_year_fraction=0.5),
        BonusEnhancedNote(put_strike=0.9, coupon_barrier=1.0, bonus_coupon=0.05,
                          maturity_year_fraction=0.5),
        BonusEnhancedNote(put_strike=0.9, coupon_barrier=1.0, bonus_coupon=0.05,
                          maturity_year_fraction=0.5, min_redemption=0.8),
        RainbowNote(put_strike=0.9, lower_call_strike=1.0, upside_participation=0.8,
                    guaranteed_flat_coupon=0.04, maturity_year_fraction=0.5),
        LeverageNote(leverage_multiple=2.0, maturity_year_fraction=0.5,
                     dividend_rate=0.04),
    ],
    ids=["ELN", "RELN", "BEN", "BEN-mr", "Rainbow", "Leverage"],
)
def test_mc_matches_analytic(product):
    """单标的产品：MC 与闭式解之差 < 3 × stderr。"""
    model = GBMModel(MARKET)
    engine = MonteCarloEngine(n_paths=100_000, batch_size=10_000, seed=77)
    r_mc = engine.price(product, model)
    r_an = product.analytic_price(MARKET)
    assert r_an is not None
    assert abs(r_mc.price - r_an) < 3 * r_mc.stderr


# ---------------------------------------------------------------------------
# 未用参数回归（优化建议 P0-2 的验证）
# ---------------------------------------------------------------------------
def test_rainbow_upside_participation_affects_price():
    """参与率必须影响价格：0.5 与 2.0 的解析价格不同且方向正确。"""
    def price_at(part):
        return RainbowNote(
            put_strike=0.9, lower_call_strike=1.0, upside_participation=part,
            guaranteed_flat_coupon=0.04, maturity_year_fraction=0.5,
        ).analytic_price(MARKET)
    p05 = price_at(0.5)
    p20 = price_at(2.0)
    assert p05 != p20
    assert p20 > p05  # 参与率越高价格越高


def test_rainbow_lower_call_strike_affects_price():
    """lower_call_strike 越高（上行更难触发）价格越低。"""
    def price_at(lcs):
        return RainbowNote(
            put_strike=0.9, lower_call_strike=lcs, upside_participation=1.0,
            guaranteed_flat_coupon=0.04, maturity_year_fraction=0.5,
        ).analytic_price(MARKET)
    assert price_at(0.95) > price_at(1.05)


def test_rainbow_validation():
    with pytest.raises(ValueError):
        RainbowNote(put_strike=1.0, lower_call_strike=0.9,
                    upside_participation=1.0, guaranteed_flat_coupon=0.0,
                    maturity_year_fraction=0.5)


# ---------------------------------------------------------------------------
# AutoCall
# ---------------------------------------------------------------------------
def test_autocall_payoff_handcrafted():
    """构造确定路径验证 AutoCall 各分支的现值。"""
    ac = AutoCallNote(
        ki_barrier=0.7, autocall_barrier=1.0, observation_frequency_days=90,
        bonus_coupon=0.05, maturity_year_fraction=0.5, min_redemption=0.8,
    )
    obs = ac.observation_year_fractions
    T = ac.maturity_year_fraction
    r = MARKET.risk_free_rate

    # 4 条路径：(敲出于 obs0, 敲出于 obs1, 到期敲入, 到期未敲出)
    rel = np.array([[
        [1.05, 0.95, 0.90, 0.99],   # obs0 相对价
        [1.10, 1.10, 0.80, 0.95],   # obs1 相对价
        [1.20, 1.20, 0.65, 0.90],   # 到期相对价
    ]])
    p = ac.payoff_paths(rel, np.ones(1), MARKET)

    expected = np.array([
        (1.0 + 0.05 * obs[0] / T) * np.exp(-r * obs[0]),          # 敲出于 obs0
        (1.0 + 0.05 * obs[1] / T) * np.exp(-r * obs[1]),          # 敲出于 obs1
        max(0.8, 0.65 / 0.7) * np.exp(-r * T),                    # 到期敲入转股
        (1.0 + 0.05) * np.exp(-r * T),                            # 到期正常赎回
    ])
    assert p == pytest.approx(expected, abs=1e-12)


def test_autocall_mc_converges():
    """AutoCall MC 与高路径数结果收敛（stderr 收敛）。"""
    ac = AutoCallNote(
        ki_barrier=0.7, autocall_barrier=1.0, observation_frequency_days=90,
        bonus_coupon=0.05, maturity_year_fraction=0.5, min_redemption=0.8,
    )
    model = GBMModel(MARKET)
    small = MonteCarloEngine(n_paths=20_000, seed=81).price(ac, model)
    large = MonteCarloEngine(n_paths=200_000, batch_size=20_000, seed=81).price(ac, model)
    assert abs(small.price - large.price) < 4 * small.stderr
    assert large.stderr < small.stderr


def test_autocall_observation_days():
    """观察日按频率生成且包含到期日。"""
    ac = AutoCallNote(
        ki_barrier=0.7, autocall_barrier=1.0, observation_frequency_days=60,
        bonus_coupon=0.05, maturity_year_fraction=0.5, min_redemption=0.8,
    )
    obs_days = ac.observation_year_fractions * 365
    assert obs_days[0] == pytest.approx(60)
    assert obs_days[-1] == pytest.approx(182.5)
    assert np.all(np.diff(obs_days) > 0)


# ---------------------------------------------------------------------------
# ELN / RELN 反解
# ---------------------------------------------------------------------------
def test_eln_strike_inversion_roundtrip():
    """反解出的 strike 定价回来等于发行价。"""
    issue_price = 0.9828
    T = 64 / 365
    strike = get_eln_strike_from_issue_price(issue_price, T, MARKET)
    eln = ELN(strike=strike, maturity_year_fraction=T)
    assert eln.analytic_price(MARKET) == pytest.approx(issue_price, abs=1e-6)


def test_reln_issue_price_roundtrip():
    """平价发行反解：ELN(反解 strike) 定价 = 发行价。"""
    strike = 1.035
    T = 64 / 365
    issue = get_reln_issue_price(strike, T, MARKET)
    reln = RELN(strike=strike, issue_price=issue, maturity_year_fraction=T)
    assert reln.analytic_price(MARKET) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 多标的与相关性
# ---------------------------------------------------------------------------
def test_multi_asset_worst_of_correlation_effect():
    """worst-of 价格随相关性上升而上升（离散度下降）。"""
    eln = ELN(strike=0.95, maturity_year_fraction=0.5)
    corr0 = MarketState(spots=[100.0, 100.0], risk_free_rate=R,
                        volatilities=[0.25, 0.25],
                        correlation=[[1.0, 0.0], [0.0, 1.0]])
    corr1 = MarketState(spots=[100.0, 100.0], risk_free_rate=R,
                        volatilities=[0.25, 0.25],
                        correlation=[[1.0, 0.99], [0.99, 1.0]])
    engine = MonteCarloEngine(n_paths=60_000, seed=91)
    model0 = GBMModel(corr0)
    model1 = GBMModel(corr1)
    p0 = engine.price(eln, model0).price
    p1 = engine.price(eln, model1).price
    # corr 高 → worst 离散度低 → put 保护更便宜 → 价格更高
    assert p1 > p0


def test_multi_asset_engine_runs():
    """多标的终端定价可运行且结果合理。"""
    eln = ELN(strike=0.95, maturity_year_fraction=0.5)
    engine = MonteCarloEngine(n_paths=20_000, seed=92)
    r = engine.price(eln, GBMModel(MARKET2))
    assert 0 < r.price <= 1.0
