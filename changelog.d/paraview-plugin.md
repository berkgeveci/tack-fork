### Added

- **Tack's filters in ParaView, at one go.** `tack.interop.paraview` makes a
  ParaView proxy of each VTK algorithm wrapper that has a panel (Contour, Clip,
  Slice, Threshold, Gradients, Elevation, Cell Quality, Extract Surface, Probe,
  Streamlines, Shrink, Feature Edges so far): its properties inferred from the
  wrapper's parameters, ParaView's names, widgets and defaults where the panel
  gives them, and a Device property. `packages/tack-vis/paraview/Tack.py` is
  the plugin that loads them all, declaring Tack (and the GPU backends'
  packages, optionally) for ParaView to install.
- **The VTK `Contour` makes normals and gradients as VTK's contour filters do**:
  `compute_normals` (from the gradient on structured grids, the surface's own
  otherwise), `compute_gradients` and `compute_scalars`, VTK's
  `ComputeNormals`, `ComputeGradients` and `ComputeScalars`.
