"""Classify every @tack.kernel in tack.dataset.algorithms by how it iterates and whether it
takes grid sizes as plain numbers. Prints a TSV: file, kernel, loop kind, loop target,
grid-number params, view methods used."""
import ast, pathlib, sys, collections

ROOT = pathlib.Path(sys.argv[1])
GRID = {"nx", "ny", "nz", "dims", "cx", "cy", "cz", "spacing", "spacings", "origin", "origins",
        "ni", "nj", "nk", "sx", "sy", "sz", "ox", "oy", "oz", "point_dims", "cell_dims",
        "extent", "extents", "width", "height", "depth", "nxy", "stride", "strides", "shape"}
METHODS = {"get_ijk", "point_index", "point_id", "entity_id", "corner_value", "index", "ijk",
           "cell_ijk", "neighbor", "neighbors"}

def is_kernel(f):
    for d in f.decorator_list:
        n = d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", None)
        if n == "kernel":
            return True
    return False

def loop_kind(f, params):
    for node in ast.walk(f):
        if isinstance(node, ast.For):
            it = node.iter
            if isinstance(it, ast.Name):
                return ("view" if it.id in params else "name"), it.id
            if isinstance(it, ast.Call):
                fn = it.func.attr if isinstance(it.func, ast.Attribute) else getattr(it.func, "id", "?")
                return fn, ast.unparse(it)[:60]
            return "other", ast.unparse(it)[:60]
    return "none", ""

rows = []
for path in sorted(ROOT.rglob("*.py")):
    tree = ast.parse(path.read_text())
    for f in ast.walk(tree):
        if isinstance(f, ast.FunctionDef) and is_kernel(f):
            params = [a.arg for a in f.args.args]
            kind, target = loop_kind(f, params)
            grid = sorted(p for p in params if p in GRID)
            used = sorted({n.attr for n in ast.walk(f) if isinstance(n, ast.Attribute) and n.attr in METHODS})
            rows.append((str(path.relative_to(ROOT)), f.name, kind, target, ",".join(grid), ",".join(used)))
for r in rows:
    print("\t".join(r))
