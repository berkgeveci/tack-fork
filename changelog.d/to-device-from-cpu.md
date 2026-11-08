### Changed

- `to_device` from the CPU to a GPU copies an array's memory straight to the
  device, not first into a new NumPy array: AhmedML's dataset (about 3 GB)
  moves to CUDA in 0.6-0.8 s, from 1.6-1.9 (jura).
