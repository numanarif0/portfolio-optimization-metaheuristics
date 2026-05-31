import numpy as np


def optimize(fobj, dim, lb, ub, n_pop=30, max_iter=200, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)

    X = rng.uniform(lb, ub, (n_pop, dim))
    fit = np.array([fobj(x) for x in X])

    g = int(np.argmin(fit))
    gbest, gbest_f = X[g].copy(), fit[g]

    mu, sigma = 25.0, 3.0
    curve = np.empty(max_iter)

    for t in range(max_iter):
        C2 = 2.0 - t / max_iter
        temp = rng.uniform(20, 35)
        p = 2.0 * (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((temp - mu) ** 2) / (2 * sigma ** 2))

        best_local = X[int(np.argmin(fit))]
        X_shade = (gbest + best_local) / 2.0
        X_food = gbest.copy()
        fit_food = gbest_f

        Xnew = X.copy()
        for i in range(n_pop):
            if temp > 30:
                if rng.random() < 0.5:
                    Xnew[i] = X[i] + C2 * rng.random() * (X_shade - X[i])
                else:
                    z = rng.integers(0, n_pop)
                    Xnew[i] = X[i] - X[z] + X_shade
            else:
                Q = 3.0 * rng.random() * (fit[i] / (fit_food - 1e-12))
                if Q > (3.0 + 1) / 2:
                    food = np.exp(-1.0 / (Q + 1e-12)) * X_food
                    Xnew[i] = X[i] + food * p * np.cos(2 * np.pi * rng.random()) \
                                   - food * p * np.sin(2 * np.pi * rng.random())
                else:
                    Xnew[i] = (X[i] - X_food) * p + p * rng.random() * X[i]

            Xnew[i] = np.clip(Xnew[i], lb, ub)

        new_fit = np.array([fobj(x) for x in Xnew])
        improved = new_fit < fit
        X[improved] = Xnew[improved]
        fit[improved] = new_fit[improved]

        if fit.min() < gbest_f:
            gbest_f = fit.min()
            gbest = X[int(np.argmin(fit))].copy()
        curve[t] = gbest_f

    return {"best_pos": gbest, "best_score": gbest_f, "curve": curve}
