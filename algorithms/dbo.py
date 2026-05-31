import numpy as np


def optimize(fobj, dim, lb, ub, n_pop=30, max_iter=200, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)

    X = rng.uniform(lb, ub, (n_pop, dim))
    fit = np.array([fobj(x) for x in X])
    X_prev = X.copy()

    gbest = X[np.argmin(fit)].copy()
    gbest_f = fit.min()

    n_roll = int(0.2 * n_pop)
    n_brood = int(0.2 * n_pop)
    n_forage = int(0.25 * n_pop)

    curve = np.empty(max_iter)
    for t in range(max_iter):
        worst = X[np.argmax(fit)]
        best_local = X[np.argmin(fit)]
        R = 1.0 - t / max_iter

        Xnew = X.copy()
        for i in range(n_pop):
            if i < n_roll:
                if rng.random() < 0.9:
                    alpha = 1 if rng.random() > 0.5 else -1
                    k, b = 0.1, 0.3
                    Xnew[i] = X[i] + alpha * k * X_prev[i] + b * np.abs(X[i] - worst)
                else:
                    theta = rng.uniform(0, np.pi)
                    if theta not in (0, np.pi / 2, np.pi):
                        Xnew[i] = X[i] + np.tan(theta) * np.abs(X[i] - X_prev[i])

            elif i < n_roll + n_brood:
                Lb = np.maximum(best_local * (1 - R), lb)
                Ub = np.minimum(best_local * (1 + R), ub)
                b1, b2 = rng.normal(size=dim), rng.normal(size=dim)
                Xnew[i] = best_local + b1 * (X[i] - Lb) + b2 * (X[i] - Ub)

            elif i < n_roll + n_brood + n_forage:
                Lbg = np.maximum(gbest * (1 - R), lb)
                Ubg = np.minimum(gbest * (1 + R), ub)
                C1 = rng.normal(size=dim)
                C2 = rng.random(dim)
                Xnew[i] = X[i] + C1 * (X[i] - Lbg) + C2 * (X[i] - Ubg)

            else:
                S = 0.5
                g = rng.normal(size=dim)
                Xnew[i] = gbest + S * g * (np.abs(X[i] - best_local) + np.abs(X[i] - gbest))

            Xnew[i] = np.clip(Xnew[i], lb, ub)

        new_fit = np.array([fobj(x) for x in Xnew])
        improved = new_fit < fit
        X_prev[improved] = X[improved]
        X[improved] = Xnew[improved]
        fit[improved] = new_fit[improved]

        if fit.min() < gbest_f:
            gbest_f = fit.min()
            gbest = X[np.argmin(fit)].copy()
        curve[t] = gbest_f

    return {"best_pos": gbest, "best_score": gbest_f, "curve": curve}
