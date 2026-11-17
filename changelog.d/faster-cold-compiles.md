### Changed

- Cold compiles verify the IR twice, after lowering and before code
  generation, rather than at every pass. The final check repeats every
  earlier one but the GPU packing's, so a pass that breaks an invariant
  still stops before code generation. The stages between run in debug mode
  or with `TACK_VERIFY_ALL` set, as the test suite sets it, and then name the
  pass; the packing's own check, on the parameters alone, runs always. On AhmedML's analysis sweep (244 kernels, cold, CPU) verification went
  from 2.6 s to 1.1 s, and compiling from 15.5 s to 13.8 s (#322).
- Cold compiles are faster in Tack's own passes, with the generated code
  unchanged: lowering learns each inlined function's source once, the
  template rewrite each method's names once, the launch bounds without
  walking the body, and copy propagation is one walk that copies only what it
  changes. AhmedML's analysis sweep (244 kernels, cold, CPU) compiles in
  11.8 s, from 15.5 (#322).
- An `if` or conditional expression whose condition integer and Boolean
  constants decide (a device class's constants, `tack.constant`s,
  literals; comparisons, `not`, `&`, `|`, `^`, `and`, `or`) is resolved
  while lowering: only the arm it selects is lowered, as C++'s
  `if constexpr`. The condition is still evaluated and typed as before
  (its operands' bounds checks in debug mode, its literals' ranges and
  types). The other arm's source is still validated, but its
  device functions are not inlined, the workgroup primitives and atomics in
  it are not required, a name bound only there is unbound (the error names
  the condition), and a variable assigned in both arms has the arm taken's
  type rather than the join of the two. Half of the
  streamlines kernel's IR on AhmedML was such arms: its cold compile went
  from 1.9 s to 1.3 s, and the analysis sweep's from 11.8 s to 10.8 s.
- Generating GPU source is faster, the source unchanged: renaming a kernel's
  bindings encodes each name where it meets it and copies nodes without
  `copy.copy`'s generic protocol (CUDA, HIP, Level Zero and Metal), and the
  MSL generator finds stores in sequential loops in one walk and checks
  workgroup participation only for kernels with collectives. MSL for the
  301 kernels of AhmedML's sweep: 0.54 s from 0.86.
- A CPU kernel's wrapper for the thread pool is compiled in the kernel's own
  module and stored with it in the disk cache, rather than JIT'd as a module
  of its own at the kernel's first fan-out in every process (2-3 ms each,
  about 0.7 s over AhmedML's sweep). First calls with a warm cache: slice
  0.33 s from 0.43, clip 0.30 from 0.42; warm calls unchanged.
