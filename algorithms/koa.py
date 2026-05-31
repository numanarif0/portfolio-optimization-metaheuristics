import numpy as np


def optimize(fobj, dim, lb, ub, n_pop=30, max_iter=200, seed=None,
             M0=0.1, lam=15.0):
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    eps = 1e-12

    X = rng.uniform(lb, ub, (n_pop, dim))
    V = rng.uniform(-1, 1, (n_pop, dim)) * (ub - lb) * 0.1
    fit = np.array([fobj(x) for x in X])

    s = int(np.argmin(fit))
    sun_pos, sun_f = X[s].copy(), fit[s]

    curve = np.empty(max_iter)
    for t in range(max_iter):
        fbest, fworst = fit.min(), fit.max()
        mu_t = M0 * np.exp(-lam * t / max_iter)
        R = np.linalg.norm(sun_pos - X, axis=1)
        Rmin, Rmax = R.min(), R.max()

        MS = np.exp(-(sun_f - fbest) / (fworst - fbest + eps))
        m = np.exp(-(fit - fbest) / (fworst - fbest + eps))

        for i in range(n_pop):
            e = rng.random()
            Tp = abs(rng.normal())
            Rn = (R[i] - Rmin) / (Rmax - Rmin + eps)

            Fg = e * mu_t * (MS * m[i]) / (Rn ** 2 + eps) + rng.random()
            a = np.cbrt(Tp ** 2 * (mu_t * (MS + m[i])) / (4 * np.pi ** 2) + eps)

            r = rng.random()
            rand_pos = rng.uniform(lb, ub)
            V[i] = r * V[i] + Fg * (sun_pos - X[i]) * a + (1 - r) * (rand_pos - X[i])

            F_flag = 1.0 if rng.random() < 0.5 else -1.0
            X_new = X[i] + F_flag * V[i] + Fg * (sun_pos - X[i])

            h = 1.0 / np.exp(t / max_iter)
            X_new = X_new + h * rng.random() * (sun_pos - X_new)

            X_new = np.clip(X_new, lb, ub)
            f_new = fobj(X_new)
            if f_new < fit[i]:
                X[i], fit[i] = X_new, f_new

        if fit.min() < sun_f:
            sun_f = fit.min()
            sun_pos = X[int(np.argmin(fit))].copy()
        curve[t] = sun_f

    return {"best_pos": sun_pos, "best_score": sun_f, "curve": curve}
