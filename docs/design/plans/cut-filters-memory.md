# Cut filters' memory

**Status: accepted.** Issue [#318](https://github.com/Kitware/tack/issues/318),
2026-11-10. Accepted in [#416](https://github.com/Kitware/tack/pull/416),
merged 2026-11-15.

## Summary

Clip holds 2.5 to 3 times VTK's memory while it runs, and on a GPU that memory
is the device's. Measured as Tack's and VTK's live allocations, the other cut
filters are already at or below VTK's peak. Most of clip's extra memory is in
how it merges points: it makes a record for every corner of every cell it
keeps and sorts them all, where VTK numbers the points it keeps through a
point map and sorts only the new points on cut edges. This plan changes that,
then removes clip's per-cell bookkeeping, and adds a count of Tack's own
allocations so every device can be measured and held to a limit.

## Where things stand

Peak memory above the input while the filter runs, on 5M-cell meshes, one
process per measurement, kernels compiled beforehand
(`benchmarks/vtk_compare/memory.py`, the M1 Max CPU, `main` at `4756ae3d`,
VTK master):

| | contour | slice | threshold | clip |
|---|---|---|---|---|
| 5M tetrahedra, VTK / Tack (MiB) | 28 / 42 | 43 / 39 | 279 / 166 | 116 / 328 |
| 5M hexahedra, VTK / Tack (MiB) | 62 / 37 | 151 / 70 | 356 / 296 | 290 / 739 |

Tack's inputs are the smaller (113 against 203 MiB for the tetrahedra, with
32-bit ids). Both systems keep outputs of about the same size (clip on the
tetrahedra: VTK 77 MiB, Tack 71).

The numbers in #318, taken on 2026-10-30, looked worse for every filter: they
were the process's peak resident memory, which on macOS keeps freed blocks
counted, and they came before clip's and contour's temporaries were halved
(`e6ac9c0b`). The benchmark now reads what the allocator has handed out and
not taken back, which counts only live allocations. Neither measure sees GPU
memory (see [D1](#d1-a-count-of-tacks-own-allocations)).

So the work is clip's. Contour on tetrahedra is 1.5 times VTK's, a smaller
matter taken last; contour on images, where it was worst (3.5 GB for
Damavand), is the flying edges path of [#401](https://github.com/Kitware/tack/pull/401).

## Where clip's memory goes

Clip works in three steps, as every filter whose output size is not known in
advance does:

1. **Count.** Each cell counts what it will output, of five kinds: cells,
   connectivity entries, point records, centroids and centroid members. The
   counts are an array of five integers per *input* cell.
2. **Scan.** The counts become each cell's starting position in the output,
   another five per input cell.
3. **Write.** Each cell writes its output at its start, including a *record*
   for each point it uses. The records are then sorted so that a point shared
   by several cells is made once (the merge).

At clip's peak (5M tetrahedra, Tack's arrays by where they were allocated):

| MiB | what | kind |
|---|---|---|
| 38 | output connectivity | output |
| 9 | output cells' sources (the input cell of each) | output |
| 12 | output cell types and offsets | output |
| 76 | the records' keys, a pair of point ids per record | merge |
| 38 | the records' slots, where each goes in the connectivity | merge |
| 38 | the sort's order (`bucket_order`) | merge |
| 78 | the sort's runs: a flag and a run number per record (`run_offsets`) | merge |
| 3 | the clip function's values at the points | input |
| **292** | | |

The 5M hexahedra, measured the same way (Tack's live arrays, by a count in
`_allocate` at the peak; the allocator's measure above, 739 MiB, also holds
blocks the allocator has not handed back):

| MiB | what | kind |
|---|---|---|
| 77 | output connectivity | output |
| 10 | output cells' sources | output |
| 13 | output cell types and offsets | output |
| 155 | the records' keys | merge |
| 77 | the records' slots | merge |
| 77 | the sort's order (`bucket_order`) | merge |
| 165 | the sort's runs (`run_offsets`) | merge |
| **574** | | |

The merge's 230 MiB (474 on the hexahedra) is the extra. It comes from one
choice: every corner a cell keeps writes a record (`keys[record] = [p, p]` in
`_clip_emit`) -- a cell kept whole for each of its corners, a cut cell for
each input point its pieces use, and a centroid for each of its members that
is an input point -- so every kept corner is sorted, not just the new points
where the clip crosses an edge. Here that is about ten million records, one
per connectivity entry, almost all of them points the input already has (20
million on the hexahedra, eight a kept cell).

The count and scan steps hold about 210 MiB of their own (five counts and
five starts per input cell), but they are released before the merge, so today
they are not the peak. Once the merge is smaller, they will be.

VTK's `vtkTableBasedClipDataSet` does neither: it numbers the input points it
keeps through a point map (one entry per input point), sorts only the edge
crossings, and counts per batch of cells, not per cell.

## The plan

### D1. A count of Tack's own allocations

Every array Tack makes adds its bytes to a count for its device, its release
takes them away, and a peak is kept (`runtime/memory_stats`, private). The
count is taken where every backend's `allocate_array` is called, in
`_allocate` and `_empty_for_mpi`, so the CPU, Metal, CUDA, HIP, Level Zero
and Warp are counted alike, and released by a finalizer on the buffer. What
it counts is the memory Tack's arrays hold:

- memory from outside (`wrap_ptr`, `array_from_ptr`, a DLPack or NumPy
  import without a copy) never passes through `_allocate`, so is not
  counted;
- a view shares its array's buffer, so is counted once, with the buffer;
- a block a backend's cache (HIP's, Metal's pool) hands to a new array is
  counted while the array holds it, and not while it sits in the cache: the
  cache's size is the backend's own matter, and `release_memory` empties it.

`reset_peak(device)` sets the peak to what is live, for a test to measure
one call. The allocator statistics the benchmark
reads see the CPU only, and GPU pools (CUDA's, HIP's buffer cache, Metal's
pool) hide what is live from the driver; this count sees every device the
same way. A test fixture reads it, so a test per cut filter can hold its peak
to a stated multiple of its output on every backend, and CI catches a
regression. It is also the accounting [#387](https://github.com/Kitware/tack/issues/387)
(spilling to host memory under a budget, 0.9) needs; it does no spilling
itself.

*Recommended.*

### D2. Clip numbers its kept points through a point map

Every use of an input point writes the point's id, as it is, and marks the
point used in a map of one entry per input point: the corners of a cell kept
whole, the input points a cut cell's pieces use (`entry < 8`), and a
centroid's members that are input points. Records are made only for
crossings, points on an edge the clip cuts, and only those are sorted. The
output's points are then numbered as today's merge numbers them: each used
input point in id order, followed by the crossings on its edges that start
at it (today's sort of `(low, high)` keys puts `(p, p)` just before
`(p, q)`). So each input point's number is a scan of the used points and of
the crossings per low point, and one pass replaces each input id by its
number, in the connectivity and in the centroids' members alike. The
connectivity's centroid placeholders (`-(centroid + 1)`, resolved after the
merge by `_centroid_entries`) are negative and an input id is not, so the
pass leaves them as they are. The output is the same bits as today.

On the tetrahedra the merge's 230 MiB falls to the map (4 MiB) and the
crossings' records, which scale with the cut, not with the kept volume; on
the hexahedra 474 MiB falls to the map (20 MiB) and the crossings'.

**Polyhedra, in a step of their own after.** `_clip_polyhedra.py` has the
same pattern: `_face_keys` writes `(a, a)` for each kept corner of each face
it clips. The map serves it the same way, with the same numbering, so it is
in this plan, as a pull request after D2's. A mixed mesh changes only with
it: `clip_mixed` keeps its whole shapes through `_keep_cells` (not
`_clip_emit`), whose points are already named once a point (`_own_keys`),
and clips the cut ones as polyhedra.

*Recommended.*

### D3. Clip counts in batches

With D2 the count and scan steps become the peak (about 210 MiB here). They
are replaced, as VTK does (`vtkBatch`):

- **Pass 1**: each batch of 256 cells counts its cells' outputs of each kind
  and writes the batch's totals: five integers per batch instead of per cell.
- **Scan** the batches' totals.
- **Pass 2**: each batch writes its cells from the batch's start, recounting
  each cell as it goes (it looks the case up anyway).

Cells are written in id order, so the output is the same bits as today. On
the CPU a batch is one thread's, walking its cells in order. On a GPU a batch
is a workgroup: its threads count their cells into shared memory, a workgroup
scan gives each thread its offset within the batch, and the batch's start is
added; without that, 5M cells would give a GPU only 20 thousand threads. A
device without workgroups (`supports_workgroups` false: the CPU, and Warp
on its CPU) takes the CPU's form, a batch a thread. The
batches' count, scan and starts are a private helper of `tack.dataset`
(`_batched.py`), so that other count-then-write filters can take it up when
a measurement says they should.

*Recommended*, with the batch size measured at 64, 256 and 1024 on the CPU
and a GPU, and fixed per backend kind.

### D4. Contour's records carry their key alone

Contour makes one record per triangle corner, carrying the point's position,
weight, source and level (40 bytes), and sorts them. Records that carry only
their edge key (8 bytes), with each unique point's position and weight
computed once after the sort from the edge's two points and the level, bring
contour on tetrahedra to about VTK's peak. Taken after D2 and D3, and after
#401 merges, since it changes `contour.py` (#389, which also did, is in).

*Recommended*, last.

### D5. Not now

- **Batching the other count-then-write filters** (threshold, extract_cells,
  extract_surface, triangulate, ...): they are at or below VTK's peak already.
  D3's helper is there when one is measured to need it.
- **Merging through a cached edge table** instead of sorting: no sort per
  call, but the table is kept for the topology's life (about 300 MiB for 5M
  tetrahedra). Worth measuring for repeated filters on one mesh (animation,
  parameter sweeps), after this plan.

## Expected

Clip's peak on the tetrahedra from 328 MiB to about its output plus the
crossings' merge, near VTK's 116; on the hexahedra from 739 MiB toward VTK's
290. Contour on tetrahedra from 42 MiB to about VTK's 28. Every output the
same bits as before.

## Validation

- Every output before and after each step, equal, on the CPU, Metal, CUDA,
  HIP and Level Zero, and on Warp where its CI job runs, over the existing
  clip and contour tests and the PyVista datasets.
- Memory by D1's count on each device, and by `memory.py` on the CPU, which
  gets a clip pair (VTK's `vtkTableBasedClipDataSet` against `td.clip`).
- A test per changed filter holding its peak, by D1, to a multiple of its
  output.

## Order

Each step is its own pull request, linking this plan:

1. D1, the allocation count, with the clip pair in `memory.py` and the memory
   tests' fixture.
2. D2, clip's point map.
3. D2 for polyhedra, `_clip_polyhedra.py`'s point map.
4. D3, clip's batches, CPU and GPU.
5. D4, contour's records, after #401 merges (#389 is in).
6. The table above again, by D1 on each device, recorded here; status: done.
