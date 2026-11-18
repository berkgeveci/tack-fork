"""Run a script in-process and report its lifetime peak physical footprint (which,
unlike RSS, counts Metal/GPU allocations on unified memory) and peak RSS.

    python measure.py script.py [args...]
"""
import ctypes
import os
import resource
import runpy
import sys
import time

libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def peak_footprint():
    buf = (ctypes.c_uint64 * 64)()          # rusage_info_v4: 16-byte uuid, then u64s
    if libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) != 0:
        return float("nan")
    fields = buf[2:]                        # skip the uuid
    return fields[28]                       # ri_lifetime_max_phys_footprint


script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
t = time.perf_counter()
try:
    runpy.run_path(script, run_name="__main__")
finally:
    print(f"[measure] wall {time.perf_counter() - t:.1f} s; peak footprint "
          f"{peak_footprint() / 2**30:.2f} GiB; peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GiB", flush=True)
