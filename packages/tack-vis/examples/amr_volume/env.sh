# Source this before running the AMR volume examples on yavin:
#   source env.sh && $PY amrwind_view.py
# The WebGPU + Slang VTK build, Dawn, slangc and wgpu-py are under $B;
# the AMReX reader is ~/Work/amrex-reader; the data is ~/Data/amrwind/plt255900.
B=/usr/local/scratch/builds/webgpu-perf
T=$HOME/Work/tack/tack-webgpu/packages
export PYTHONPATH=$B/vtk-build/lib/python3.13/site-packages:$T/tack-core/src:$T/tack-vis/src:$B/wgpu-py:$HOME/Work/amrex-reader
export DYLD_LIBRARY_PATH=$B/dawn-install/lib
export SLANGC=$B/slang-2026.22/bin/slangc
PY=$HOME/Work/tack/tack/.venv/bin/python
cd $T/tack-vis/examples/amr_volume
