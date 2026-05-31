# Investment Portfolio Optimization with Recent Metaheuristic Algorithms

Optimization Algorithms course project. The problem is **Markowitz mean–variance
portfolio optimization**: distribute capital across assets to **maximize the
risk-adjusted return (Sharpe ratio)**. Five optimization algorithms published in
the last 2–3 years are implemented and compared against the classic PSO baseline
on **real stock-market data**.

## Algorithms
| Code | Algorithm | Year | Role |
|------|-----------|------|------|
| PSO  | Particle Swarm Optimization | 1995 | baseline |
| DBO  | Dung Beetle Optimizer | 2023 | recent |
| COA  | Crayfish Optimization Algorithm | 2023 | recent |
| KOA  | Kepler Optimization Algorithm | 2023 | recent |
| CPO  | Crested Porcupine Optimizer | 2024 | recent |
| SBOA | Secretary Bird Optimization Algorithm | 2024 | recent |

## Project structure
```
portfolio_optimization/
├── data_loader.py      # real stock data -> mu (returns), Sigma (covariance)
├── objective.py        # Sharpe-ratio fitness with constraints (PortfolioObjective)
├── algorithms/         # one file per optimizer, common optimize() interface
│   ├── pso.py  dbo.py  coa.py  koa.py  cpo.py  sboa.py
│   └── __init__.py     # ALGORITHMS registry
├── experiment.py       # 30 independent runs, metrics, Wilcoxon test
├── plots.py            # convergence / boxplot / efficient frontier / weights
├── main.py             # end-to-end runner
└── results/            # CSV tables + PNG figures (generated)
```

## Setup & run
```bash
pip install -r requirements.txt
python main.py                       # 30 runs x 200 iterations (default)
python main.py --runs 10 --iter 100  # quicker
python main.py --cardinality 5       # at most 5 assets (extra constraint)
```

## Data
12 sector-diversified U.S. large-cap stocks (AAPL, MSFT, NVDA, JPM, GS, JNJ,
PFE, KO, PG, WMT, XOM, CAT), ~6 years of daily prices via `yfinance`. The first
successful download is cached to `results/prices.csv` for offline reruns.

## Outputs (in `results/`)
- `summary.csv` — Best/Mean/Worst/Std Sharpe, time, NFE per algorithm
- `sharpe_per_run.csv` — raw Sharpe of every run (for the boxplot)
- `wilcoxon.csv` — statistical significance vs. the best algorithm
- `best_weights.csv` — best portfolio weights found by each algorithm
- `convergence.png`, `boxplot.png`, `efficient_frontier.png`, `weights.png`
