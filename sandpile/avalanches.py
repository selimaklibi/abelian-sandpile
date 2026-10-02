"""Self-organised criticality: drive a finite sandpile with a sink and record avalanches.

Grains are dropped one at a time on uniformly random sites of an L x L grid
whose border is a sink. After a transient the pile reaches its stationary
(recurrent) regime, where avalanche sizes follow an approximate power law
P(S >= s) ~ s^{-(tau - 1)} up to a cut-off growing with L.
"""
from __future__ import annotations

import numpy as np


def drive(L: int, n_grains: int, rng: np.random.Generator, burn_in: int | None = None):
    """Return (sizes, areas, final_heights) for `n_grains` drops after `burn_in` drops.

    size = number of topplings in the avalanche, area = number of distinct toppled sites.
    Uses an explicit stack on Python lists (avalanches are mostly small, so this beats
    vectorised sweeps).
    """
    burn_in = 3 * L * L if burn_in is None else burn_in
    h = [[0] * L for _ in range(L)]
    total = burn_in + n_grains
    xs = rng.integers(0, L, size=total)
    ys = rng.integers(0, L, size=total)
    sizes = np.zeros(n_grains, dtype=np.int64)
    areas = np.zeros(n_grains, dtype=np.int64)
    for t in range(total):
        i, j = int(xs[t]), int(ys[t])
        row = h[i]
        row[j] += 1
        if row[j] < 4:
            continue
        size = 0
        toppled = set()
        stack = [(i, j)]
        while stack:
            a, b = stack.pop()
            ha = h[a]
            if ha[b] < 4:
                continue
            k = ha[b] // 4
            ha[b] -= 4 * k
            size += k
            toppled.add((a, b))
            if a > 0:
                h[a - 1][b] += k
                if h[a - 1][b] >= 4: stack.append((a - 1, b))
            if a < L - 1:
                h[a + 1][b] += k
                if h[a + 1][b] >= 4: stack.append((a + 1, b))
            if b > 0:
                ha[b - 1] += k
                if ha[b - 1] >= 4: stack.append((a, b - 1))
            if b < L - 1:
                ha[b + 1] += k
                if ha[b + 1] >= 4: stack.append((a, b + 1))
        if t >= burn_in:
            sizes[t - burn_in] = size
            areas[t - burn_in] = len(toppled)
    return sizes, areas, np.array(h)


def tail_exponent_mle(x: np.ndarray, x_min: int, x_max: int | None = None) -> tuple[float, float, int]:
    """Discrete power-law exponent by maximum likelihood (Clauset, Shalizi & Newman 2009, eq. 3.7).

    Uses observations in [x_min, x_max]; returns (alpha_hat, std_error, n_used),
    where P(X = x) ~ x^{-alpha}. The upper truncation guards against the finite-size cut-off
    (the estimator then ignores truncation, which is acceptable when x_max >> x_min).
    """
    x = np.asarray(x)
    sel = x[(x >= x_min) & ((x <= x_max) if x_max else True)]
    n = sel.size
    alpha = 1.0 + n / np.sum(np.log(sel / (x_min - 0.5)))
    return float(alpha), float((alpha - 1) / np.sqrt(n)), int(n)


def mean_avalanche_size_exact(L: int) -> float:
    """Dhar's theorem: in the stationary regime the expected number of topplings caused by a
    grain dropped at x is G(x, .) summed, with G = (-Delta_Dirichlet)^{-1}. Averaging over a
    uniform drop site gives  E[S] = (1/L^2) 1' G 1  (grows like L^2)."""
    from scipy.sparse import diags, identity, kron
    from scipy.sparse.linalg import spsolve

    T = diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(L, L))
    minus_lap = (kron(T, identity(L)) + kron(identity(L), T)).tocsc()
    return float(spsolve(minus_lap, np.ones(L * L)).mean())
