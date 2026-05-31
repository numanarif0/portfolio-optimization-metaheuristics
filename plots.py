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
    plt.xlabel("İterasyon"); plt.ylabel("En iyi Sharpe oranı")
    plt.title("Ortalama Yakınsama Eğrileri (30 koşu ortalaması)")
    plt.legend(); plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "convergence.png"), dpi=150)
    plt.close()


def plot_boxplot():
    df = pd.read_csv(os.path.join(RESULTS, "sharpe_per_run.csv"))
    plt.figure(figsize=(8, 5))
    plt.boxplot([df[c] for c in df.columns], labels=df.columns, showmeans=True)
    plt.ylabel("Sharpe oranı"); plt.title("30 Koşunun Sharpe Dağılımı (Kararlılık)")
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
    plt.colorbar(sc, label="Sharpe oranı")

    wdf = pd.read_csv(os.path.join(RESULTS, "best_weights.csv"), index_col=0)
    markers = ["*", "o", "s", "D", "^", "P"]
    for (name, w), mk in zip(wdf.items(), markers):
        obj = PortfolioObjective(mu, Sigma, rf=rf)
        r, v, s = obj.stats(w.values)
        plt.scatter(v, r, marker=mk, s=180, edgecolors="black",
                    label=f"{name} (S={s:.2f})", zorder=5)

    plt.xlabel("Yıllık Risk (Volatilite)"); plt.ylabel("Yıllık Beklenen Getiri")
    plt.title("Etkin Sınır (Efficient Frontier) ve Algoritma Çözümleri")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "efficient_frontier.png"), dpi=150)
    plt.close()


def plot_weights():
    wdf = pd.read_csv(os.path.join(RESULTS, "best_weights.csv"), index_col=0)
    ax = wdf.T.plot(kind="bar", stacked=True, figsize=(9, 5), colormap="tab20")
    ax.set_ylabel("Ağırlık"); ax.set_xlabel("Algoritma")
    ax.set_title("En İyi Portföy Ağırlıkları (Algoritma Bazında)")
    ax.legend(title="Hisse", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "weights.png"), dpi=150)
    plt.close()


def make_all():
    plot_convergence(); plot_boxplot(); plot_efficient_frontier(); plot_weights()
    print("Figürler kaydedildi: convergence.png, boxplot.png, efficient_frontier.png, weights.png")


if __name__ == "__main__":
    make_all()
