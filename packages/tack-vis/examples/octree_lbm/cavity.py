# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Lid-driven cavity at Re 100 on refined quadtrees, against Ghia et al. (1982)
Table I and against uniform meshes."""
import sys
import time

import numpy as np

import tack
from tack.dataset import octree_lbm as lb

tack.set_default_device("cpu")
N, U, Re = 32, 0.1, 100.0
tau = 3 * U * N / Re + 0.5
GHIA_Y = np.array([1.0, .9766, .9688, .9609, .9531, .8516, .7344, .6172, .5, .4531, .2813, .1719, .1016, .0703, .0625, .0547, 0.0])
GHIA_U = np.array([1.0, .84123, .78871, .73722, .68717, .23151, .00332, -.13641, -.20581, -.21090, -.15662, -.10150, -.06434, -.04775, -.04192, -.03717, 0.0])

def centreline(s):
    rho, ux, uy = s.moments(); c, side = s.centres(); keep = s.computational()
    touch = keep & (np.abs(np.abs(c[:, 0] - N / 2) - side / 2) < 1e-9)
    ys = np.round(c[touch, 1], 9); ys_u = np.unique(ys)
    u = np.array([ux[touch][ys == y].mean() for y in ys_u]) / U
    return np.concatenate([[0], ys_u / N, [1]]), np.concatenate([[0], u, [1]])

def run(name, refine, finest, most=40000):
    s = lb.LBM2D(lb.lbm_mesh((N, N), refine, finest, periodic=(False, False)), tau, lid=U)
    s.initialize(1.0, 0.0, 0.0); m0 = s.mass()
    t0 = time.perf_counter(); prev = None; steps = 0
    while steps < most:
        s.step(1000); steps += 1000
        y, u = centreline(s)
        if prev is not None and np.abs(u - prev).max() < 1e-5:
            break
        prev = u
    dt = time.perf_counter() - t0
    mine = np.interp(GHIA_Y, y, u)
    cells = int(s.computational().sum())
    print(f"{name:26s} {cells:6d} cells, {steps:5d} steps in {dt:5.1f} s ({steps * s.updates_per_step() / dt / 1e6:4.1f} MLUPS): "
          f"largest |u - Ghia| {np.abs(mine - GHIA_U).max():.4f}, rms {np.sqrt(np.mean((mine - GHIA_U) ** 2)):.4f}; mass change {(s.mass() - m0) / m0:.1e}")
    return s, y, u

band = lambda x, y, h: (y > 24) if h > 0.75 else (y > 29)
results = {}
results["uniform 32"] = run("uniform 32x32", lambda x, y, h: False, 0)
results["uniform 64"] = run("uniform 64x64", lambda x, y, h: True, 1)
results["refined1"] = run("32, one level under lid", lambda x, y, h: y > 24, 1)
results["refined"] = run("32, two levels under lid", band, 2)
results["uniform 128"] = run("uniform 128x128", lambda x, y, h: True, 2)
np.savez(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", **{k.replace(" ", "_"): np.stack([v[1], v[2]]) for k, v in results.items()})
