import numpy as np


def optimize(fobj, dim, lb, ub, n_pop=30, max_iter=200, seed=None,
             w_max=0.9, w_min=0.2, c1=2.0, c2=2.0):
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    v_max = 0.2 * (ub - lb)

    X = rng.uniform(lb, ub, (n_pop, dim))
    V = rng.uniform(-v_max, v_max, (n_pop, dim))
    fit = np.array([fobj(x) for x in X])

    pbest = X.copy()
    pbest_f = fit.copy()
    g = int(np.argmin(fit))
    gbest = X[g].copy()
    gbest_f = fit[g]

    curve = np.empty(max_iter)
    for t in range(max_iter):
        w = w_max - t * (w_max - w_min) / max_iter
        r1, r2 = rng.random((n_pop, dim)), rng.random((n_pop, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -v_max, v_max)
        X = np.clip(X + V, lb, ub)

        fit = np.array([fobj(x) for x in X])
        improved = fit < pbest_f
        pbest[improved] = X[improved]
        pbest_f[improved] = fit[improved]

        g = int(np.argmin(pbest_f))
        if pbest_f[g] < gbest_f:
            gbest_f = pbest_f[g]
            gbest = pbest[g].copy()
        curve[t] = gbest_f

    return {"best_pos": gbest, "best_score": gbest_f, "curve": curve}
