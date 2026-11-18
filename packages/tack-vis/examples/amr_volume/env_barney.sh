# Source this before running amrwind_barney.py:
#   source env_barney.sh && $PY amrwind_barney.py
# Barney (CPU backend, ~/Work/Anari/barney with the AMR BVH patch) and pynari
# (~/Work/Anari/pynari) are built against the ANARI SDK in ~/Work/Anari/install-sdk
# and installed in $B/install. VTK is the venv's pip VTK (OpenGL): no WebGPU stack.
B=/usr/local/scratch/builds/barney
T=$HOME/Work/tack/tack-webgpu/packages
export PYTHONPATH=$B/install/lib:$HOME/Work/amrex-reader
export DYLD_LIBRARY_PATH=$B/install/lib:$HOME/Work/Anari/install-sdk/lib
export ANARI_LIBRARY=barney
PY=$HOME/Work/tack/tack/.venv/bin/python
cd $T/tack-vis/examples/amr_volume
