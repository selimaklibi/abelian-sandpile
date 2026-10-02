import numpy as np
import pytest

from sandpile import (stabilize, stabilize_sequential, point_source, visited_set, boundary,
                      is_connected, THRESHOLD)


def _center(res, G):
    """Crop a Result to a common window centred on the pile, for comparisons."""
    f, u = res.final, res.odometer
    nz = np.argwhere(f + u > 0)
    lo, hi = nz.min(0), nz.max(0) + 1
    return f[lo[0]:hi[0], lo[1]:hi[1]], u[lo[0]:hi[0], lo[1]:hi[1]]


def _laplacian(u):
    p = np.pad(u, 1)
    return p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4 * u


@pytest.mark.parametrize("seed", range(5))
def test_abelian_property_random_orders(seed):
    """Final configuration AND odometer do not depend on the toppling order."""
    rng = np.random.default_rng(seed)
    config = rng.integers(0, 8, size=(5, 5))
    ref_f, ref_u = _center(stabilize(config), config.sum())
    for order in ("random", "fifo"):
        f, u = _center(stabilize_sequential(config, rng, order=order), config.sum())
        np.testing.assert_array_equal(f, ref_f)
        np.testing.assert_array_equal(u, ref_u)


@pytest.mark.parametrize("G", [4, 17, 100, 300])
def test_point_source_order_independence_and_step_count(G):
    rng = np.random.default_rng(G)
    par = stabilize(point_source(G))
    seq = stabilize_sequential(point_source(G), rng)
    assert par.topplings == seq.sweeps == seq.topplings  # l = l' : same number of avalanches


@pytest.mark.parametrize("G", [1000, 5000])
def test_final_state_is_stable_conserves_mass_and_matches_odometer(G):
    res = stabilize(point_source(G))
    assert res.final.max() < THRESHOLD and res.final.min() >= 0
    assert res.final.sum() == G
    initial = np.zeros_like(res.final)
    c = np.array(res.final.shape) // 2
    initial[tuple(c)] = G
    np.testing.assert_array_equal(res.final, initial + _laplacian(res.odometer))


@pytest.mark.parametrize("G", [1000, 5000, 20000])
def test_total_topplings_equal_second_moment_over_four(G):
    """Exact identity on Z^2: sum_x u(x) = (1/4) sum_x |x|^2 eta_final(x) for a point source."""
    res = stabilize(point_source(G))
    c = np.array(res.final.shape) // 2
    i, j = np.indices(res.final.shape)
    r2 = (i - c[0]) ** 2 + (j - c[1]) ** 2
    assert 4 * res.topplings == int((r2 * res.final).sum())


@pytest.mark.parametrize("G", [50, 1000, 10000])
def test_geometry_of_visited_set(G):
    res = stabilize(point_source(G))
    C = visited_set(res)
    assert is_connected(C)
    assert boundary(C).sum() <= G
    assert C.sum() <= G
    assert C.sum() <= 4 * res.topplings + 1
    # every boundary site never toppled, hence still holds >= 1 grain
    assert (res.final[boundary(C)] >= 1).all()
    # C is inside [-G, G]^2 around the origin
    c = np.array(res.final.shape) // 2
    pts = np.argwhere(C) - c
    assert np.abs(pts).max() <= G


@pytest.mark.parametrize("G", [1000, 30000])
def test_dihedral_symmetry_of_point_source(G):
    f = stabilize(point_source(G)).final
    for g in (np.rot90(f), np.fliplr(f), f.T):
        np.testing.assert_array_equal(g, f)


def test_sink_loses_grains_and_stabilises():
    res = stabilize(np.full((20, 20), 7), sink=True)
    assert res.final.max() < THRESHOLD
    assert res.final.sum() < 7 * 400
