#!/bin/zsh
# Run cylinder_view.py on VTK's WebGPU build, Tack from the study worktree (the
# solver runs on the CPU, so the venv needs no Metal).
#   cylinder_view.sh [--re 100] [--diameter 8] [--steps 8] [--check OUTDIR]
W=~/Work/tack/tack-octree-filters
V=/usr/local/scratch/builds/VTK/webgpu-python-volume-mapper
B=/usr/local/scratch/builds/webgpu-perf
PYTHONPATH=$V/lib/python3.13/site-packages:$W/packages/tack-core/src:$W/packages/tack-vis/src \
DYLD_LIBRARY_PATH=$B/dawn-install/lib:$B/slangkit-install/lib \
exec $W/.venv/bin/python -u ${0:A:h}/cylinder_view.py "$@"
