r"""Stabilisation of sandpile configurations.

Convention (standard BTW): a site is *unstable* when it holds at least 4 grains;
toppling removes 4 grains and sends one to each of its 4 neighbours, i.e.
eta <- eta + A^{x} with the discrete Laplacian stencil [[0,1,0],[1,-4,1],[0,1,0]].

Two stabilisers are provided:

* :func:`stabilize` - vectorised NumPy. Every unstable site topples
  floor(eta/4) times at once. By the abelian property this legal schedule
  reaches the same final configuration with the same odometer as any other.
* :func:`stabilize_sequential` - one site at a time, in random or FIFO order.
  Slow; used in the tests to *check* the abelian property empirically.

The odometer u(x) is the number of times x toppled. Final state and odometer are
linked by eta_final = eta_0 + Laplacian(u), which the tests verify.

On Z^2 (``sink=False``) the grid is grown until no grain ever touches its
border, so the result is exactly the infinite-lattice one. With ``sink=True``
grains falling off the border are lost (finite grid with a sink vertex).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

THRESHOLD = 4


@dataclass
class Result:
    final: np.ndarray       # stable configuration
    odometer: np.ndarray    # number of topplings at each site
    sweeps: int             # number of parallel update rounds

    @property
    def topplings(self) -> int:
        return int(self.odometer.sum())


def _laplacian_add(target: np.ndarray, q: np.ndarray) -> None:
    """target += Laplacian(q) on the interior (q is zero on the border)."""
    target -= 4 * q
    target[1:, :] += q[:-1, :]
    target[:-1, :] += q[1:, :]
    target[:, 1:] += q[:, :-1]
    target[:, :-1] += q[:, 1:]


def _stabilize_fixed(h: np.ndarray, sink: bool) -> Result:
    h = h.astype(np.int64, copy=True)
    u = np.zeros_like(h)
    sweeps = 0
    while True:
        q = h // THRESHOLD
        if sink:
            q[0, :] = q[-1, :] = q[:, 0] = q[:, -1] = 0  # border cells are the sink
            h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = 0
        if not q.any():
            break
        _laplacian_add(h, q)
        u += q
        sweeps += 1
    return Result(h, u, sweeps)


def stabilize(config: np.ndarray, sink: bool = False) -> Result:
    """Stabilise `config`.

    sink=False: infinite lattice Z^2 (the array is a window, padded as needed).
    sink=True : finite grid whose outer ring is the sink.
    """
    if sink:
        padded = np.pad(config, 1)
        res = _stabilize_fixed(padded, sink=True)
        return Result(res.final[1:-1, 1:-1], res.odometer[1:-1, 1:-1], res.sweeps)

    pad = 2
    while True:
        padded = np.pad(config, pad)
        res = _stabilize_fixed(padded, sink=True)
        # Nothing reached the sink ring => identical to the run on Z^2.
        if res.final.sum() == config.sum() and not res.odometer[1:-1, 1:-1][[0, -1], :].any() \
                and not res.odometer[1:-1, 1:-1][:, [0, -1]].any():
            return res if pad == 0 else Result(res.final, res.odometer, res.sweeps)
        pad *= 2


def stabilize_sequential(config: np.ndarray, rng: np.random.Generator | None = None,
                         order: str = "random") -> Result:
    """Topple one unstable site at a time (on Z^2, via generous padding).

    order="random": pick a uniformly random unstable site at each step.
    order="fifo"  : process unstable sites in a queue.
    """
    rng = rng or np.random.default_rng()
    G = int(config.sum())
    pad = G + 2  # C_n is contained in [-G, G]^2 (see README)
    h = np.pad(config.astype(np.int64), pad)
    u = np.zeros_like(h)
    unstable = {tuple(x) for x in np.argwhere(h >= THRESHOLD)}
    queue = deque(sorted(unstable))
    steps = 0
    while unstable:
        if order == "random":
            x = list(unstable)[rng.integers(len(unstable))]
        else:
            x = queue.popleft()
            if x not in unstable:
                continue
        i, j = x
        h[i, j] -= 4
        u[i, j] += 1
        steps += 1
        if h[i, j] < THRESHOLD:
            unstable.discard(x)
        for y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            h[y] += 1
            if h[y] >= THRESHOLD and y not in unstable:
                unstable.add(y)
                queue.append(y)
        if order == "fifo" and x in unstable:
            queue.append(x)
    return Result(h, u, steps)


def point_source(n_grains: int) -> np.ndarray:
    """Configuration n * 1_0 on a 1x1 window (padded automatically by stabilize)."""
    return np.array([[n_grains]], dtype=np.int64)


def visited_set(res: Result) -> np.ndarray:
    """Sites that received at least one grain = sites with a toppling neighbour (or that toppled)."""
    t = res.odometer > 0
    v = t.copy()
    v[1:, :] |= t[:-1, :]
    v[:-1, :] |= t[1:, :]
    v[:, 1:] |= t[:, :-1]
    v[:, :-1] |= t[:, 1:]
    return v


def boundary(mask: np.ndarray) -> np.ndarray:
    """x in dC iff x in C and some 4-neighbour of x is not in C."""
    m = np.pad(mask, 1)
    inner = m[2:, 1:-1] & m[:-2, 1:-1] & m[1:-1, 2:] & m[1:-1, :-2]
    return mask & ~inner


def is_connected(mask: np.ndarray) -> bool:
    """4-connectivity of a boolean mask (BFS)."""
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return True
    seen = np.zeros_like(mask)
    dq = deque([tuple(pts[0])])
    seen[tuple(pts[0])] = True
    count = 1
    H, W = mask.shape
    while dq:
        i, j = dq.popleft()
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < H and 0 <= b < W and mask[a, b] and not seen[a, b]:
                seen[a, b] = True
                count += 1
                dq.append((a, b))
    return count == len(pts)
