# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

import sys
import time

import numpy as np

import tack
from tack.dataset import octree_lbm as lb

tack.set_default_device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
def tg(c, u0=0.03):
    k = 2 * np.pi / N; x, y = c[:, 0], c[:, 1]
    return (1 - 3 * u0 * u0 / 4 * (np.cos(2 * k * x) + np.cos(2 * k * y)), -u0 * np.cos(k * x) * np.sin(k * y), u0 * np.sin(k * x) * np.cos(k * y))
sq = lambda x, y, h: abs(x - N/2) < N/4 and abs(y - N/2) < N/4
two = lambda x, y, h: sq(x, y, h) if h > 0.75 else (abs(x - N/2) < N/8 and abs(y - N/2) < N/8)
for name, r, fl in (("uniform", lambda x, y, h: False, 0), ("uniform, one finer", lambda x, y, h: True, 1),
                    ("square, two levels", sq, 1), ("squares, three levels", two, 2)):
    t = time.perf_counter(); m = lb.lbm_mesh((N, N), r, fl); s = lb.LBM2D(m, 0.8); build = time.perf_counter() - t
    c, _ = s.centres(); s.initialize(*tg(c)); s.step(3); tack.sync()
    steps = 50
    t = time.perf_counter(); s.step(steps); tack.sync(); dt = (time.perf_counter() - t) / steps
    cells = int(s.computational().sum())
    print(f"{name:22s} {cells:7d} cells ({np.bincount(s.role, minlength=4)[2]:6d} ghosts), mesh {build:5.1f} s: "
          f"{1000 * dt:6.2f} ms a step, {s.updates_per_step() / dt / 1e6:6.1f} MLUPS")
