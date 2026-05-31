import numpy as np
from math import gamma as _gamma


def _levy(dim, rng, beta=1.5):
    sigma = (_gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, dim)
    v = rng.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta))


def optimize(fobj, dim, lb, ub, n_pop=30, max_iter=200, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)

    X = rng.uniform(lb, ub, (n_pop, dim))
    fit = np.array([fobj(x) for x in X])
    b = int(np.argmin(fit))
    best_pos, best_f = X[b].copy(), fit[b]

    curve = np.empty(max_iter)
    for t in range(max_iter):
        ratio = t / max_iter

        for i in range(n_pop):
            if ratio < 1 / 3:
                r1, r2 = rng.integers(0, n_pop), rng.integers(0, n_pop)
                X_new = X[i] + (X[r1] - X[r2]) * rng.random(dim)
            elif ratio < 2 / 3:
                RB = rng.normal(size=dim)
                X_new = best_pos + np.exp((ratio) ** 4) * (RB - 0.5) * (best_pos - X[i])
            else:
                RL = 0.5 * _levy(dim, rng)
                X_new = best_pos + (1 - ratio) ** (2 * ratio) * X[i] * RL

            X_new = np.clip(X_new, lb, ub)
            f_new = fobj(X_new)
            if f_new < fit[i]:
                X[i], fit[i] = X_new, f_new
                if f_new < best_f:
                    best_f, best_pos = f_new, X_new.copy()

            if rng.random() < 0.5:
                RB = rng.normal(size=dim)
                X_esc = best_pos + (2 * RB - 1) * (1 - ratio) ** 2 * X[i]
            else:
                K = round(1 + rng.random())
                r = rng.integers(0, n_pop)
                X_esc = X[i] + rng.random() * (X[r] - K * X[i])

            X_esc = np.clip(X_esc, lb, ub)
            f_esc = fobj(X_esc)
            if f_esc < fit[i]:
                X[i], fit[i] = X_esc, f_esc
                if f_esc < best_f:
                    best_f, best_pos = f_esc, X_esc.copy()

        curve[t] = best_f

    return {"best_pos": best_pos, "best_score": best_f, "curve": curve}
