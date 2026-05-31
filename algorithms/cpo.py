import numpy as np


def optimize(fobj, dim, lb, ub, n_pop=30, max_iter=200, seed=None,
             T_cycles=2, alpha=0.2, tf=0.8):
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    eps = 1e-12
    N_min = max(2, int(0.5 * n_pop))

    X = rng.uniform(lb, ub, (n_pop, dim))
    fit = np.array([fobj(x) for x in X])
    gb = int(np.argmin(fit))
    best_pos, best_f = X[gb].copy(), fit[gb]

    cycle_len = max(1, int(np.ceil(max_iter / T_cycles)))
    curve = np.empty(max_iter)

    for t in range(max_iter):
        N_t = int(N_min + (n_pop - N_min) * (1 - (t % cycle_len) / cycle_len))
        N_t = max(N_min, N_t)

        gamma = 2 * rng.random() * (1 - t / max_iter) ** (t / max_iter)
        S = np.exp(fit / (fit.sum() + eps))

        for i in range(N_t):
            r1, r2, r3 = (rng.integers(0, N_t) for _ in range(3))
            delta = 1.0 if rng.random() >= 0.5 else -1.0
            U1 = (rng.random(dim) > 0.5).astype(float)
            y = (X[i] + X[rng.integers(0, N_t)]) / 2.0

            if rng.random() < tf:
                if rng.random() < 0.5:
                    tau1 = rng.normal(size=dim)
                    X_new = X[i] + tau1 * np.abs(2 * rng.random() * best_pos - y)
                else:
                    tau3 = rng.random()
                    X_new = (1 - U1) * X[i] + U1 * (y + tau3 * (X[r1] - X[r2]))
            else:
                if rng.random() < 0.5:
                    tau3 = rng.random()
                    X_new = (1 - U1) * X[i] + U1 * (
                        X[r1] + S[i] * (X[r2] - X[r3]) - tau3 * delta * gamma * S[i])
                else:
                    tau4, tau5 = rng.random(), rng.random()
                    Fi = rng.random() * np.exp(-fit[i] / (fit.sum() + eps)) * (X[r1] - X[i])
                    X_new = best_pos + (alpha * (1 - tau4) + tau4) * (delta * best_pos - X[i]) \
                        - tau5 * delta * gamma * Fi

            X_new = np.clip(X_new, lb, ub)
            f_new = fobj(X_new)
            if f_new < fit[i]:
                X[i], fit[i] = X_new, f_new
                if f_new < best_f:
                    best_f, best_pos = f_new, X_new.copy()

        curve[t] = best_f

    return {"best_pos": best_pos, "best_score": best_f, "curve": curve}
