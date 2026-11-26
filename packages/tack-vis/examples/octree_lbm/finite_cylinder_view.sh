#!/bin/zsh
# A finite cylinder in 3D: sphere_view.py --body cylinder (see sphere_view.sh).
#   finite_cylinder_view.sh [--length 2] [--re 200] [--diameter 8] [--device metal] ...
exec ${0:A:h}/sphere_view.sh --body cylinder "$@"
