#!/bin/zsh
# Run mpi_render.py on N MPI ranks: Tack from the octree study's worktree, VTK's
# WebGPU build (the AMR volume modules need it), and in one `uv run` the main
# venv (mpi4py, built against Homebrew's Open MPI) with Metal's bindings and
# adios2 added for the run only, so every rank's `python` is that one.
#   mpi_render.sh [-n 2] [PIECES.npz OCTREE.bp] [--check] [--out picture.png] [--step 0.5] ...
# By default: yA31 step 15159's pieces and its playback copy, on the T7.
W=~/Work/tack/tack-octree-filters
V=/usr/local/scratch/builds/VTK/webgpu-python-volume-mapper
B=/usr/local/scratch/builds/webgpu-perf
T=/Volumes/T7/deepwaterimpact
n=2
if [[ $1 == -n ]]; then n=$2; shift 2; fi
pieces=$T/pieces/p15159.npz
octree=$T/bp_preview/L2/pv_insitu_15159.bp
if [[ $# -ge 2 && $1 != --* ]]; then pieces=$1; octree=$2; shift 2; fi
export PYTHONPATH=$V/lib/python3.13/site-packages:$W/packages/tack-core/src:$W/packages/tack-vis/src
export DYLD_LIBRARY_PATH=$B/dawn-install/lib:$B/slangkit-install/lib
cd ${0:A:h} && exec uv run --project ~/Work/tack/tack --no-sync \
    --with pyobjc-framework-Metal --with pyobjc-framework-MetalKit --with adios2 \
    mpiexec -x PYTHONPATH -x DYLD_LIBRARY_PATH -n $n python mpi_render.py $pieces $octree "$@"
