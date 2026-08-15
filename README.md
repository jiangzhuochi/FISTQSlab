# FISTQSlab

小型结构化衍生品定价库：Black-Scholes 闭式解 + 分块 Monte Carlo 引擎。
支持 ELN / RELN / BEN / AutoCall / Rainbow / Leverage Note 等结构化产品，
以及欧式期权（call / put / digital）与其希腊字母。

本版本是对旧版（两套 BS、两代 Monte Carlo、路径全量读入内存）的系统性重构，
详见 [`优化报告.md`](优化报告.md)。

## 支持的产品

| 产品 | 类 | payoff 特征 |
| --- | --- | --- |
| 欧式认购/认沽 | `EuropeanCallOption` / `EuropeanPutOption` | `max(S_T − K, 0)` |
| 现金或无期权 | `DigitalCallOption` / `DigitalPutOption` | `cash·1{条件}` |
| ELN | `ELN` | `1 − (strike−worst)⁺/strike` |
| RELN | `RELN` | `(strike+issue−1) − (strike−worst)⁺` |
| 增利票据 | `BonusEnhancedNote` | 保本/非保本两版 |
| 自动赎回票据 | `AutoCallNote` | 观察日敲出/到期敲入 |
| worst-of 彩虹 | `RainbowNote` | 参与率上行 + 下行保护 |
| 杠杆票据 | `LeverageNote` | 线性杠杆 |

## 安装

```bash
pip install -e .        # 或 poetry install
pip install pytest pytest-cov ruff   # 开发依赖
```

## Quick Start

```python
from fistqslab import (
    MarketState, GBMModel, MonteCarloEngine, AnalyticEngine, ELN,
)

market = MarketState(spots=[100.0], risk_free_rate=0.02, volatilities=0.25)
eln = ELN(strike=0.95, maturity_year_fraction=0.5)

model = GBMModel(market)
mc = MonteCarloEngine(n_paths=100_000, batch_size=20_000, seed=42).price(eln, model)
print(mc.price, mc.stderr, mc.ci_low, mc.ci_high)   # 价格 / 标准误 / 95% CI

analytic = AnalyticEngine().price(eln, market)       # 闭式对照
```

## 利率 / 波动率 / 时间单位约定

- **所有利率为年化连续复利**，折现因子恒为 `exp(−r·T)`；
- **所有波动率为年化**；
- **时间参数一律为 year fraction（年）**：`T = year_fraction(days) = days/365`；
  观察日同理（Act/365）；
- 票息为**整个期限的总票息**（示例中按期限折算）；
- 多标的相关性通过 `MarketState(correlation=...)` 的 Cholesky 分解施加；
- 随机数一律 `np.random.default_rng(seed)`，相同 `seed` 结果逐位可复现，
  也是 Common Random Numbers（CRN）Greeks 的基础；
- `n_paths` 为独立抽样数，`antithetic=True`（默认）时有效路径数为 `2×n_paths`。

## Monte Carlo 精度说明

- 分块生成：内存与 `batch_size` 成正比，与路径总数无关
  （100 万路径 ≈ 数十 MB）；
- 结果含 `stderr` 与 95% 置信区间（MC 结果没有 stderr 是不完整的）；
- 仅看终值的产品直接模拟 `S_T`（O(路径) 而非 O(路径×时间步)）；
- 路径依赖产品（AutoCall）只在观察日布点。

## 架构

```
src/fistqslab/
├── market/     MarketState（spot/rate/vol/corr）+ day_count
├── models/     GBMModel（分块模拟）+ Black-Scholes 解析
├── products/   payoff 定义（不持有市场参数）
├── engines/    MonteCarloEngine（分块+累加器）/ AnalyticEngine
├── risk/       GreeksCalculator（有限差分 + CRN）
├── results.py  PricingResult（price/stderr/CI/seed）
└── app.py      Flask API（POST /euro_option_bs）
```

调用模式：`engine.price(product, model)`——同一个引擎可定价所有产品。

## 测试

```bash
pytest            # 62 项：BS 基准/parity/Greeks FD、MC vs 闭式、
                  # 手造路径 payoff、CRN、相关性、1M 路径内存冒烟
ruff check src tests
```

## 已知限制

- 多标的 worst-of 闭式解未实现（Monte Carlo 为主）；
- Greeks 采用有限差分（CRN），未实现 pathwise / likelihood-ratio；
- 未实现 Sobol / quasi-MC 与 control variates；
- 观察日按自然日等间隔生成，未接入交易日历。