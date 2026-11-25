### Added

- `tack.cpp` (a prototype, CPU backend only): a kernel or `@tack.func` calls a
  C++ function of a `tack.cpp.Library` (headers or source text) on scalars.
  The call's types instantiate a template; clang (`TACK_CLANG`, else `clang++`
  on the path, needed only where such a kernel is compiled) writes LLVM bitcode
  that is linked into the kernel before it is optimized, so the library's code
  is inlined and vectorized with the loop and stored with the variant in the
  disk cache.
