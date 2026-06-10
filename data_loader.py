import os
import numpy as np
import pandas as pd

TRADING_DAYS = 252

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA",
    "JPM", "GS",
    "JNJ", "PFE",
    "KO", "PG", "WMT",
    "XOM",
    "CAT",
]

CACHE_PATH = os.path.join(os.path.dirname(__file__), "results", "prices.csv")


def download_prices(tickers=None, start="2019-01-01", end="2024-12-31"):
    tickers = tickers or DEFAULT_TICKERS
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        prices = raw["Close"] if "Close" in raw else raw
        prices = prices.dropna(how="all").dropna(axis=1, how="any")
        prices = prices[[t for t in tickers if t in prices.columns]]
        if len(prices) > 50:
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            prices.to_csv(CACHE_PATH)
            return prices
        raise RuntimeError("Downloaded data is insufficient.")
    except Exception as exc:
        if os.path.exists(CACHE_PATH):
            print(f"[data_loader] Online download failed ({exc}); using cache.")
            return pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        raise


def prepare_inputs(tickers=None, start="2019-01-01", end="2024-12-31", rf_annual=0.02):
    prices = download_prices(tickers, start, end)
    returns = prices.pct_change().dropna()

    mu = returns.mean().values * TRADING_DAYS
    Sigma = returns.cov().values * TRADING_DAYS

    meta = {
        "tickers": list(prices.columns),
        "n_assets": prices.shape[1],
        "returns_daily": returns,
        "prices": prices,
        "rf": rf_annual,
        "mu": mu,
        "Sigma": Sigma,
    }
    return mu, Sigma, meta


if __name__ == "__main__":
    mu, Sigma, meta = prepare_inputs()
    print("Num. assets   :", meta["n_assets"])
    print("Tickers       :", meta["tickers"])
    print("Observations  :", len(meta["returns_daily"]))
    print("\nAnnualized expected returns (mu):")
    for t, m in zip(meta["tickers"], mu):
        print(f"  {t:5s}: {m:6.2%}")
    print("\nAnnualized volatilities (sqrt(diag Sigma)):")
    for t, s in zip(meta["tickers"], np.sqrt(np.diag(Sigma))):
        print(f"  {t:5s}: {s:6.2%}")
