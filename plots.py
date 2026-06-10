import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import prepare_inputs
from objective import PortfolioObjective

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def plot_convergence():
    data = np.load(os.path.join(RESULTS, "curves.npz"))
    plt.figure(figsize=(8, 5))
    for name in data.files:
        plt.plot(data[name], linewidth=2, label=name)
    plt.xlabel("Iteration"); plt.ylabel("Best Sharpe ratio")
    plt.title("Mean Convergence Curves (average of 30 runs)")
    plt.legend(); plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "convergence.png"), dpi=150)
    plt.close()


def plot_boxplot():
    df = pd.read_csv(os.path.join(RESULTS, "sharpe_per_run.csv"))
    plt.figure(figsize=(8, 5))
    plt.boxplot([df[c] for c in df.columns], showmeans=True)
    plt.xticks(range(1, len(df.columns) + 1), df.columns)
    plt.ylabel("Sharpe ratio"); plt.title("Sharpe Ratio Distribution over 30 Runs (Robustness)")
    plt.grid(True, ls="--", alpha=0.5, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "boxplot.png"), dpi=150)
    plt.close()


def plot_efficient_frontier(n_random=20000, rf=0.02):
    mu, Sigma, meta = prepare_inputs(rf_annual=rf)
    dim = meta["n_assets"]
    rng = np.random.default_rng(42)

    W = rng.random((n_random, dim))
    W /= W.sum(axis=1, keepdims=True)
    rets = W @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, Sigma, W))
    sharpes = (rets - rf) / vols

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(vols, rets, c=sharpes, cmap="viridis", s=4, alpha=0.4)
    plt.colorbar(sc, label="Sharpe ratio")

    wdf = pd.read_csv(os.path.join(RESULTS, "best_weights.csv"), index_col=0)
    markers = ["*", "o", "s", "D", "^", "P"]
    for (name, w), mk in zip(wdf.items(), markers):
        obj = PortfolioObjective(mu, Sigma, rf=rf)
        r, v, s = obj.stats(w.values)
        plt.scatter(v, r, marker=mk, s=180, edgecolors="black",
                    label=f"{name} (S={s:.2f})", zorder=5)

    plt.xlabel("Annual Risk (Volatility)"); plt.ylabel("Annual Expected Return")
    plt.title("Efficient Frontier and Algorithm Solutions")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "efficient_frontier.png"), dpi=150)
    plt.close()


def plot_weights():
    wdf = pd.read_csv(os.path.join(RESULTS, "best_weights.csv"), index_col=0)
    ax = wdf.T.plot(kind="bar", stacked=True, figsize=(9, 5), colormap="tab20")
    ax.set_ylabel("Weight"); ax.set_xlabel("Algorithm")
    ax.set_title("Best Portfolio Weights per Algorithm")
    ax.legend(title="Asset", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "weights.png"), dpi=150)
    plt.close()


def make_all():
    plot_convergence(); plot_boxplot(); plot_efficient_frontier(); plot_weights()
    print("Figures saved: convergence.png, boxplot.png, efficient_frontier.png, weights.png")


if __name__ == "__main__":
    make_all()
