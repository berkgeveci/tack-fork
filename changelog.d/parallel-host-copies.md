### Changed

- Copying NumPy data into CPU arrays (`tack.array`, `tack.copyto`, a dataset's
  conversion from VTK) is done by up to eight threads for arrays of 32 MiB and
  more, and a conversion `same_kind` casting allows (64-bit ids into 32, f64
  into f32) happens in that copy rather than in a converted copy made first.
  On jura a GB copies at about 13 GB/s, from 3.5-5, and AhmedML's 21.5M cells
  convert in 4.1-4.2 s, from 5.3-5.7. `TACK_CPU_THREADS` bounds the copy's
  threads too.
