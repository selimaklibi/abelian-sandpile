import numpy as np
import pytest

from sandpile import drive, tail_exponent_mle


def test_drive_reaches_stationary_density():
    sizes, areas, h = drive(32, 5000, np.random.default_rng(1))
    assert h.max() < 4
    assert (areas <= sizes).all()
    assert 2.0 < h.mean() < 2.2          # stationary density ~ 2.125 on Z^2 (lower near the sink)
    assert (sizes == 0).mean() > 0.3      # most grains trigger nothing


def test_mle_recovers_known_exponent():
    rng = np.random.default_rng(0)
    alpha = 2.5
    x = rng.zipf(alpha, size=200_000)
    est, se, _ = tail_exponent_mle(x, x_min=10)
    assert est == pytest.approx(alpha, abs=4 * se + 0.02)


def test_mean_avalanche_size_matches_dhar_theorem():
    """Simulation vs. exact Green's-function prediction (L = 16: E[S] = 11.338...)."""
    from sandpile import mean_avalanche_size_exact
    L = 16
    sizes, _, _ = drive(L, 100_000, np.random.default_rng(7))
    exact = mean_avalanche_size_exact(L)
    se = sizes.std() / np.sqrt(sizes.size)
    assert abs(sizes.mean() - exact) < 5 * se
