"""Reproduce the figures and numbers of the README.   python scripts/make_figures.py [--quick]"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sandpile import stabilize, point_source, visited_set, drive, tail_exponent_mle, mean_avalanche_size_exact

QUICK = "--quick" in sys.argv
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#0b0b0b", "ink2": "#52514e",
     "muted": "#898781", "grid": "#e6e5e1", "surface": "#fcfcfb"}
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["muted"],
    "axes.labelcolor": C["ink2"], "xtick.color": C["muted"], "ytick.color": C["muted"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.titleweight": "bold",
    "axes.titlesize": 11, "lines.linewidth": 2, "legend.frameon": False,
})


def fig_point_source():
    G = 2 ** (14 if QUICK else 17)
    res = stabilize(point_source(G))
    f = res.final
    nz = np.argwhere(f > 0)
    lo, hi = nz.min(0), nz.max(0) + 1
    f = f[lo[0]:hi[0], lo[1]:hi[1]]
    cmap = ListedColormap([C["surface"], "#cde2fb", "#5598e7", "#104281"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(f, cmap=cmap, interpolation="nearest", vmin=0, vmax=3)
    ax.set_axis_off()
    ax.set_title(f"Stable configuration of {G:,} grains dropped at the origin of Z²\n"
                 f"{res.topplings:,} topplings  ·  light → dark = 0, 1, 2, 3 grains", fontsize=10)
    fig.savefig(OUT / "point_source.png")
    print(f"G = {G}: topplings = {res.topplings}, diameter = {hi[0]-lo[0]}, density = {G/visited_set(res).sum():.3f}")


def fig_scaling():
    Gs = 2 ** np.arange(6, 15 if QUICK else 17)
    tops, radii = [], []
    for G in Gs:
        r = stabilize(point_source(int(G)))
        tops.append(r.topplings)
        radii.append(np.sqrt(visited_set(r).sum() / np.pi))
    tops, radii = np.array(tops), np.array(radii)
    b_t = np.polyfit(np.log(Gs), np.log(tops), 1)[0]
    b_r = np.polyfit(np.log(Gs), np.log(radii), 1)[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].loglog(Gs, tops, "o", color=C["blue"], ms=6, label=f"simulation (slope {b_t:.3f})")
    axes[0].loglog(Gs, Gs.astype(float) ** 2 / (8 * np.pi * 2.125), "--", color=C["orange"], lw=1.5,
                   label=r"$G^2 / (8\pi\rho)$, $\rho = 2.125$")
    axes[0].set(xlabel="grains G", ylabel="total topplings", title="Topplings scale as G²")
    axes[1].loglog(Gs, radii, "o", color=C["blue"], ms=6, label=f"simulation (slope {b_r:.3f})")
    axes[1].loglog(Gs, np.sqrt(Gs / (np.pi * 2.125)), "--", color=C["orange"], lw=1.5,
                   label=r"$\sqrt{G / (\pi\rho)}$")
    axes[1].set(xlabel="grains G", ylabel="radius of visited set", title="Radius scales as √G")
    for ax in axes:
        ax.legend(fontsize=8.5)
    fig.savefig(OUT / "scaling.png")
    print(f"scaling slopes: topplings {b_t:.4f}, radius {b_r:.4f}")


def fig_avalanches():
    cfg = [(32, 200_000), (64, 200_000), (128, 50_000)] if not QUICK else [(16, 20_000), (32, 20_000)]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    rows = []
    for (L, n), col in zip(cfg, RAMP[1:]):
        s, _, _ = drive(L, n, np.random.default_rng(L))
        sim_mean = s.mean()
        s = s[s > 0]
        xs = np.sort(s)
        ccdf = 1.0 - np.arange(xs.size) / xs.size
        ax.loglog(xs, ccdf, color=col, label=f"L = {L}")
        tau, se, _ = tail_exponent_mle(s, 10, L * L // 4)
        rows.append((L, tau, se, sim_mean, mean_avalanche_size_exact(L)))
    ax.set(xlabel="avalanche size s (topplings)", ylabel="P(S ≥ s | S > 0)",
           title="Self-organised criticality: heavy-tailed avalanche sizes, cut-off growing with L")
    ax.legend()
    fig.savefig(OUT / "avalanches.png")
    print("L   tau_hat  se      mean(S) sim   mean(S) exact (Dhar)")
    for L, tau, se, m, e in rows:
        print(f"{L:<4}{tau:.3f}    {se:.3f}   {m:9.2f}     {e:9.2f}")


if __name__ == "__main__":
    fig_point_source()
    fig_scaling()
    fig_avalanches()
