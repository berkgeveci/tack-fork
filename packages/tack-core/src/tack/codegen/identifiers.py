# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Collision-free identifiers shared by code generation and entry lookup.

Every IR binding is encoded, rather than testing vendor keyword lists. The
variable and kernel namespaces are disjoint from each other and from emitted
helpers/temporaries. ASCII Python identifiers retain a readable spelling;
other names use UTF-8 hex in a separate namespace. Both encodings are injective,
including user names that already resemble an encoded name.
"""

from tack.lang.ir_names import NAME_ARRAYS, NAME_LIST_ARRAYS
from tack.lang.ir_traversal import transform_ir


def _encode(name: str) -> str:
    if name.isascii() and name.isidentifier():
        # C++ reserves identifiers containing double underscores, even away
        # from the beginning. Escape '_' and the escape character itself so
        # Python's dunder names never enter that implementation namespace.
        return 'a_' + name.replace('Z', 'Z1').replace('_', 'Z0')
    return 'u_' + name.encode('utf-8').hex()


def kernel_entry_name(name: str) -> str:
    """Encode the original IR function name, exactly once at each use.

    LLVM also uses this spelling: llvmlite's JIT symbol lookup requires ASCII,
    and a user entry name must not collide with a declared libm function.
    """
    return 'tack_kernel_' + _encode(name)


def gpu_variable_name(name: str) -> str:
    return 'tack_var_' + _encode(name)


def rename_gpu_bindings(ir_func):
    """Return a codegen-only copy; canonical IR and dispatch metadata stay intact.

    Function names remain original: source emission and runtime lookup both
    call kernel_entry_name on that spelling. Parameter order and binding indices
    are unchanged. Intrinsics, attributes, types and metadata are not bindings.
    """
    # The encoding is injective, so each name is encoded where it is first met
    # rather than after a walk collecting them all.
    names = {}

    def encoded(name):
        found = names.get(name)
        if found is None:
            found = names[name] = gpu_variable_name(name)
        return found

    def rename(node):
        for attr in NAME_ARRAYS.get(type(node), ()):
            name = getattr(node, attr)
            if name is not None:
                setattr(node, attr, encoded(name))
        for attr in NAME_LIST_ARRAYS.get(type(node), ()):
            if getattr(node, attr):
                setattr(node, attr, [encoded(name) for name in getattr(node, attr)])
        return node

    return transform_ir(ir_func, rename, copy_nodes=True)
