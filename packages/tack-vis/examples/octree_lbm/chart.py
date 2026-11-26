# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(sys.argv[1])
GY = np.array([1.0, .9766, .9688, .9609, .9531, .8516, .7344, .6172, .5, .4531, .2813, .1719, .1016, .0703, .0625, .0547, 0.0])
GU = np.array([1.0, .84123, .78871, .73722, .68717, .23151, .00332, -.13641, -.20581, -.21090, -.15662, -.10150, -.06434, -.04775, -.04192, -.03717, 0.0])
fig, ax = plt.subplots(figsize=(6, 5.5), dpi=150)
for key, label, colour, style in (("uniform_32", "uniform 32×32 (1,024 cells)", "#9aa5b1", "--"),
                                  ("refined", "32×32 + two levels under the lid (2,944 cells)", "#2a6fdb", "-"),
                                  ("uniform_128", "uniform 128×128 (16,384 cells)", "#e07b39", ":")):
    y, u = d[key]
    ax.plot(u, y, style, color=colour, lw=1.8, label=label)
ax.plot(GU, GY, "o", mfc="none", mec="#222", ms=6, label="Ghia et al. 1982")
ax.set_xlabel("u / lid speed at x = 0.5"); ax.set_ylabel("y")
ax.set_title("Lid-driven cavity, Re 100: vertical centreline", fontsize=11)
ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(sys.argv[2])
