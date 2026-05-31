import os
import time
import numpy as np
import pandas as pd
from scipy.stats import ranksums

from data_loader import prepare_inputs
from objective import PortfolioObjective
from algorithms import ALGORITHMS

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def run_experiment(n_runs=30, n_pop=30, max_iter=200, rf=0.02,
                   cardinality=None, seed_base=0):
    mu, Sigma, meta = prepare_inputs(rf_annual=rf)
    dim = meta["n_assets"]
    lb, ub = np.zeros(dim), np.ones(dim)
    os.makedirs(RESULTS, exist_ok=True)

    sharpe_runs = {name: np.empty(n_runs) for name in ALGORITHMS}
    times = {name: np.empty(n_runs) for name in ALGORITHMS}
    nfes = {name: np.empty(n_runs) for name in ALGORITHMS}
    curves = {name: np.zeros(max_iter) for name in ALGORITHMS}
    best_weights = {}
    best_sharpe = {name: -np.inf for name in ALGORITHMS}

    print(f"Deney: {n_runs} koşu x {len(ALGORITHMS)} algoritma "
          f"(pop={n_pop}, iter={max_iter}, varlık={dim})\n")

    for name, (fn, year, desc) in ALGORITHMS.items():
        for r in range(n_runs):
            obj = PortfolioObjective(mu, Sigma, rf=rf, cardinality=cardinality)
            t0 = time.perf_counter()
            res = fn(obj, dim, lb, ub, n_pop=n_pop, max_iter=max_iter, seed=seed_base + r)
            times[name][r] = time.perf_counter() - t0
            nfes[name][r] = obj.eval_count

            ret, vol, sh = obj.stats(res["best_pos"])
            sharpe_runs[name][r] = sh
            curves[name] += (-res["curve"])
            if sh > best_sharpe[name]:
                best_sharpe[name] = sh
                best_weights[name] = obj.to_weights(res["best_pos"])

        curves[name] /= n_runs
        s = sharpe_runs[name]
        print(f"{name:5s} ({year}) | Best={s.max():.4f}  Mean={s.mean():.4f}  "
              f"Worst={s.min():.4f}  Std={s.std():.4f}  "
              f"t={times[name].mean():.2f}s  NFE={int(nfes[name].mean())}")

    rows = []
    for name, (fn, year, desc) in ALGORITHMS.items():
        s = sharpe_runs[name]
        rows.append({
            "Algorithm": name, "Year": year, "Description": desc,
            "Best": s.max(), "Mean": s.mean(), "Worst": s.min(),
            "Std": s.std(), "Time_s": times[name].mean(),
            "NFE": int(nfes[name].mean()),
        })
    summary = pd.DataFrame(rows).sort_values("Mean", ascending=False)

    champ = summary.iloc[0]["Algorithm"]
    wil = []
    for name in ALGORITHMS:
        if name == champ:
            wil.append({"Algorithm": name, "vs_champion": champ, "p_value": np.nan, "significant": "-"})
            continue
        _, p = ranksums(sharpe_runs[champ], sharpe_runs[name])
        wil.append({"Algorithm": name, "vs_champion": champ, "p_value": p,
                    "significant": "Evet" if p < 0.05 else "Hayir"})
    wilcoxon = pd.DataFrame(wil)

    wdf = pd.DataFrame(best_weights, index=meta["tickers"])

    summary.to_csv(os.path.join(RESULTS, "summary.csv"), index=False)
    pd.DataFrame(sharpe_runs).to_csv(os.path.join(RESULTS, "sharpe_per_run.csv"), index=False)
    wilcoxon.to_csv(os.path.join(RESULTS, "wilcoxon.csv"), index=False)
    wdf.to_csv(os.path.join(RESULTS, "best_weights.csv"))
    np.savez(os.path.join(RESULTS, "curves.npz"), **curves)

    print("\n=== ÖZET (ortalama Sharpe'a göre sıralı) ===")
    print(summary.to_string(index=False))
    print("\n=== Wilcoxon (şampiyon:", champ, ") ===")
    print(wilcoxon.to_string(index=False))
    print("\nSonuçlar 'results/' klasörüne kaydedildi.")

    return {"summary": summary, "sharpe_runs": sharpe_runs, "curves": curves,
            "weights": wdf, "wilcoxon": wilcoxon, "meta": meta}


if __name__ == "__main__":
    run_experiment()
