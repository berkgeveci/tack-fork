"""A field over many boxes, as kernels read it: ``field.at(b, k)``, box ``b``'s value
``k``. Two layouts, one kernel source (each class compiles its own variant):

``Joined``: the boxes' values in one array, box ``b`` from ``starts[b]`` -- one buffer
(or views of one, end to end, as PR #458 shares the AMReX reader's). Any device.

``Table``: each box its own array, reached through a table of their addresses
(``tack.indirect_load``) -- separately allocated arrays, no copy. CPU only.

``field_of(parts, layout)`` makes either from the boxes' arrays.
"""
import numpy as np

import tack
from tack import arrays


@tack.device_class
class Joined:
    def __init__(self, values, starts):
        self.values = values
        self.starts = starts

    @tack.func
    def at(self, b, k):
        return self.values[self.starts[b] + k]


@tack.device_class
class Table:
    def __init__(self, parts):
        hosts = [_host(p) for p in parts]
        dtype = hosts[0].dtype
        if any(h.dtype != dtype for h in hosts):
            raise ValueError("Table: the boxes' arrays must share one dtype")
        self.table = tack.array(np.fromiter((h.ctypes.data for h in hosts), np.int64,
                                            len(hosts)))
        self.like = tack.array(np.zeros(1, dtype))
        self._parts = parts               # the table's addresses stay valid with them

    @tack.func
    def at(self, b, k):
        return tack.indirect_load(self.table, b, k, self.like)


def _host(part):
    """The NumPy array a CPU array's values are, no copy (a view's, for a view)."""
    if isinstance(part, np.ndarray):
        data = part
    else:
        if getattr(part._buffer, "backend_name", None) != "cpu":
            raise ValueError("Table: CPU arrays only")
        data = part._buffer._data
    if not data.flags.c_contiguous:
        raise ValueError("Table: contiguous arrays only")
    return data


def field_of(parts, layout="joined", what="field"):
    """The boxes' arrays ``parts`` as a ``Joined`` (they must be views of one array, end
    to end) or a ``Table`` (any arrays on the CPU)."""
    if layout == "table":
        return Table(parts)
    whole = arrays._joined_view(parts)
    if whole is None:
        raise ValueError(f"{what}: the boxes' arrays are not views of one array, end to end")
    sizes = [p.size for p in parts]
    return Joined(whole, tack.array(np.concatenate([[0], np.cumsum(sizes)[:-1]])
                                    .astype(np.int64)))
