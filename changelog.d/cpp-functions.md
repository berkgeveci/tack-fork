### Added

- `tack.cpp` (a prototype, CPU backend only): a kernel or `@tack.func` calls a
  C++ function of a `tack.cpp.Library` (headers or source text) on scalars and
  arrays (`tack.cpp.ptr`: an array or a `tack.local_array` as a pointer, `const`
  unless `writes=True`, which Tack counts as a store). A `@tack.device_class`
  object passes for a template parameter the C++ calls methods on
  (`tack.cpp.obj(method=(result, args))`): the C++ receives a struct whose
  methods call the object's `@tack.func` methods, inlined into it.
  The call's types instantiate a template; clang (`TACK_CLANG`, else `clang++`
  on the path, needed only where such a kernel is compiled) writes LLVM bitcode
  that is linked into the kernel before it is optimized, so the library's code
  is inlined and vectorized with the loop and stored with the variant in the
  disk cache. A library is parsed once a process, into a precompiled header
  its instantiations are compiled against.
