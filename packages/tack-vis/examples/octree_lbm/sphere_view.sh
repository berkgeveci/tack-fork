#!/bin/zsh
# Run sphere_view.py on VTK's WebGPU build, Tack from the study worktree. With
# --device metal, uv adds the Metal bindings for this run only (an overlay; the
# worktree's venv stays without them).
#   sphere_view.sh [--re 300] [--diameter 8] [--steps 4] [--start 0] [--device cpu]
#                  [--check OUTDIR]
W=~/Work/tack/tack-octree-filters
V=/usr/local/scratch/builds/VTK/webgpu-python-volume-mapper
B=/usr/local/scratch/builds/webgpu-perf
export PYTHONPATH=$V/lib/python3.13/site-packages:$W/packages/tack-core/src:$W/packages/tack-vis/src
export DYLD_LIBRARY_PATH=$B/dawn-install/lib:$B/slangkit-install/lib
script=${0:A:h}/sphere_view.py
if [[ " $* " == *" metal "* || " $* " == *"--device=metal"* ]]; then
    cd $W && exec uv run --no-sync --with pyobjc-framework-Metal --with pyobjc-framework-MetalKit \
        python -u $script "$@"
fi
exec $W/.venv/bin/python -u $script "$@"
