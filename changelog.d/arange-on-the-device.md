### Changed

- `tack.arange` of integers (i32 or i64, 65,536 values or more) is filled by a
  kernel where the array lives, not made by NumPy on the host and copied: 22M
  ids on CUDA in under a millisecond, from about 0.1 s, and on the CPU in
  parallel. Its values and type are NumPy's as before; floats keep NumPy's way.
