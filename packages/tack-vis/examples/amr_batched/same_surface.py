"""Whether two meshes (.npz from tack_isosurface.py) hold the same triangles, up to the
order of the triangles and of each triangle's vertices: compared by positions.

    python same_surface.py A.npz B.npz
"""
import sys

import numpy as np


def canonical(path):
    m = np.load(path)
    tri = m["points"][m["triangles"]]                    # (n, 3, 3)
    v = tri.view([("x", tri.dtype), ("y", tri.dtype), ("z", tri.dtype)]).reshape(-1, 3)
    v = np.sort(v, axis=1)                               # each triangle's vertices in order
    flat = v.view(tri.dtype).reshape(len(tri), 9)
    order = np.lexsort(flat.T[::-1])
    return flat[order]


a, b = canonical(sys.argv[1]), canonical(sys.argv[2])
print(f"{len(a)} vs {len(b)} triangles; same set: "
      f"{a.shape == b.shape and np.array_equal(a, b)}")
