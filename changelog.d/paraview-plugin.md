### Added

- **Tack's filters in ParaView, at one go.** `tack.interop.paraview` makes a
  ParaView proxy of every VTK algorithm wrapper but `Interpolate` and
  `Composite` (78 filters): its properties inferred from the wrapper's
  parameters, ParaView's names, widgets and defaults where its panel gives them
  -- ParaView's implicit functions with their widgets for Clip, Slice, Extract
  Cells By Region, Extract Points and Implicit Values, its Transform for
  Transform, Resample To Image's sampling grid for Deposit and Voxelize -- and a
  Device property. `packages/tack-vis/paraview/Tack.py` is the plugin that
  loads them all, declaring Tack (and the GPU backends' packages, optionally)
  for ParaView to install.
- **The VTK `Contour` makes normals and gradients as VTK's contour filters do**:
  `compute_normals` (from the gradient on structured grids, the surface's own
  otherwise), `compute_gradients` and `compute_scalars`, VTK's
  `ComputeNormals`, `ComputeGradients` and `ComputeScalars`.

### Fixed

- **A `vtkPlane`'s `Offset` and `AxisAligned`** are kept where a VTK wrapper takes
  it as an implicit function (they were ignored).
- **A wrapper whose new field takes the field's name** (`ValuesAtCenters`,
  `Project`) takes the name of the field `SetInputArrayToProcess` chose, instead of
  asking for one.
