### Changed

- CUDA kernels are compiled by NVRTC to the device's own code (CUBIN) where NVRTC
  knows its architecture, and the disk cache keeps that, not PTX: a process that
  finds its kernels in the cache loads them without the driver compiling them.
  The AhmedML sweep's 313 kernels from a warm Tack cache and a cold driver cache
  (CI, a fresh node, a container) in 2.1 s, from 10.6 (jura); a cold compile
  costs the same, NVRTC doing the work the driver did. A device NVRTC does not
  know still gets PTX.
