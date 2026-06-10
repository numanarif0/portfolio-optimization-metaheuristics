import os
import time
import numpy as np
import pandas as pd

from data_loader import prepare_inputs
from objective import PortfolioObjective
from algorithms import ALGORITHMS

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def tune(algos=("CPO", "SBOA", "DBO"), pops=(10, 20, 30, 50),
         iters=(100, 200, 300), n_runs=15, rf=0.02, seed_base=100):
    mu, Sigma, meta = prepare_inputs(rf_annual=rf)
    dim = meta["n_assets"]
    lb, ub = np.zeros(dim), np.ones(dim)

    rows = []
    for algo in algos:
        fn = ALGORITHMS[algo][0]
        print(f"Fine-tuning: {algo}  (pop x iter, {n_runs} runs)")
        for p in pops:
            for it in iters:
                sh = np.empty(n_runs)
                t0 = time.perf_counter()
                for r in range(n_runs):
                    obj = PortfolioObjective(mu, Sigma, rf=rf)
                    res = fn(obj, dim, lb, ub, n_pop=p, max_iter=it, seed=seed_base + r)
                    sh[r] = obj.stats(res["best_pos"])[2]
                dt = (time.perf_counter() - t0) / n_runs
                rows.append({"algo": algo, "pop": p, "iter": it,
                             "mean_sharpe": sh.mean(), "std_sharpe": sh.std(),
                             "time_s": dt})
                print(f"  pop={p:3d} iter={it:3d} | Sharpe={sh.mean():.4f} "
                      f"(std {sh.std():.4f})  t={dt:.2f}s")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "tuning.csv"), index=False)

    for algo in algos:
        sub = df[df["algo"] == algo]
        pivot = sub.pivot(index="pop", columns="iter", values="mean_sharpe")
        print(f"\n=== Mean Sharpe for {algo} (rows=pop, cols=iter) ===")
        print(pivot.round(4).to_string())
    print("\nResult: results/tuning.csv")
    return df


if __name__ == "__main__":
    tune()
