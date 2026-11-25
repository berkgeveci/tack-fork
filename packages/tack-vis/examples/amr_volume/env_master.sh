# Source this before running tack_view.py on yavin:
#   source env_master.sh && $PY tack_view.py --check /tmp/out
# VTK master with vtkPythonWebGPUVolumeMapper (~/Work/VTK/webgpu-python-volume-mapper) is
# built under $V; Dawn and SlangKit are webgpu-perf's. $PY has Tack's CPU backend;
# $PYM also has the Metal bindings.
V=/usr/local/scratch/builds/VTK/webgpu-python-volume-mapper
B=/usr/local/scratch/builds/webgpu-perf
T=$HOME/Work/tack/tack-webgpu/packages
export PYTHONPATH=$V/lib/python3.13/site-packages:$T/tack-core/src:$T/tack-vis/src:$HOME/Work/amrex-reader
export DYLD_LIBRARY_PATH=$B/dawn-install/lib:$B/slangkit-install/lib
export SLANGC=$B/slang-2026.22/bin/slangc
PY=$HOME/Work/tack/tack/.venv/bin/python
PYM=$HOME/Work/uv-venvs/3.13/bin/python
cd $T/tack-vis/examples/amr_volume
