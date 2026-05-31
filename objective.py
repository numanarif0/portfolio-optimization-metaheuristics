import numpy as np


class PortfolioObjective:

    def __init__(self, mu, Sigma, rf=0.02, cardinality=None, penalty=1e3):
        self.mu = np.asarray(mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.rf = rf
        self.n = len(mu)
        self.cardinality = cardinality
        self.penalty = penalty
        self.eval_count = 0

    def to_weights(self, x):
        x = np.clip(np.asarray(x, dtype=float), 0.0, None)
        s = x.sum()
        if s <= 1e-12:
            return np.ones(self.n) / self.n
        w = x / s

        if self.cardinality is not None and self.cardinality < self.n:
            keep = np.argsort(w)[-self.cardinality:]
            mask = np.zeros_like(w)
            mask[keep] = w[keep]
            w = mask / mask.sum()
        return w

    def stats(self, x):
        w = self.to_weights(x)
        ret = float(w @ self.mu)
        vol = float(np.sqrt(max(w @ self.Sigma @ w, 1e-12)))
        sharpe = (ret - self.rf) / vol
        return ret, vol, sharpe

    def __call__(self, x):
        self.eval_count += 1
        w = self.to_weights(x)
        ret = w @ self.mu
        vol = np.sqrt(max(w @ self.Sigma @ w, 1e-12))
        sharpe = (ret - self.rf) / vol

        penalty = 0.0
        if self.cardinality is not None:
            active = int(np.sum(w > 1e-4))
            if active > self.cardinality:
                penalty += self.penalty * (active - self.cardinality)

        return -sharpe + penalty
