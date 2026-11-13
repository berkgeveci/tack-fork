# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tack's surfaces drawn by VTK's own WebGPU mapper (``vtkPolyDataMapper``), through the
typed Slang hooks of VTK's Slang branch (SlangKit's ``IShaderHooks``).

One actor per launch group (shape, order). Its polydata is the group's faces as
template triangles (``indexed_template``: each face subdivided ``level`` times a side),
with the template's (u, v) as texture coordinates. Its shader property carries a Slang
module: Tack's own ``position``/Jacobian and ``value`` exported as Slang functions
(``slang_gen``, the dataset's arrays packed in one buffer per function) and a hook type
``TackSurface`` holding those two buffers as ``StructuredBuffer<uint>`` fields
(``vtkWebGPUShaderProperty.SetSlangBuffer``, added on the VTK branch for this):

- ``vertex``: with ``place`` on, the point is put where Tack's geometry says, its normal
  from the Jacobian (the geometry per vertex, on the GPU); off, the polydata's points
  and normals are used (Tack evaluated them once on the host: the "cached" form);
- ``color``: the field evaluated at the fragment's interpolated (u, v) in its face
  (``cellId / per_face``), through a colour map: exact per pixel;
- ``output``: probes for checking (``probe`` 1-4: the value, u, v, the face, as 24-bit
  numbers in the pixel's colour).

Instanced (``instanced=True``): the polydata is one face's template, drawn once per face
(``vtkWebGPUPolyDataMapper.SetNumberOfInstances``, added on the VTK branch); the face is
the instance (``instanceId``), the vertex hook places every point, and the mapper's
bounds are given (``SetDrawnBounds``), from Tack's geometry at a coarse template. No
polydata of the surface: a geometry step changes Tack's buffer and the bounds alone.

VTK keeps everything else: camera, lights, depth, the actor's property, picking.
"""

import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "slang_surfaces"),
                os.path.join(HERE, "..", "surface_bench")]

import vtkmodules.vtkCommonDataModel  # noqa: E402,F401
import vtkmodules.vtkRenderingCore as rr  # noqa: E402
import vtkmodules.vtkRenderingWebGPU  # noqa: E402,F401
from scenes import indexed_template  # noqa: E402
from shaders import _geometry_at, _one_cell_arrays, _value_at  # noqa: E402
from slang_gen import export, joined  # noqa: E402
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkPoints, vtkUnsignedIntArray  # noqa: E402
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData  # noqa: E402

import tack  # noqa: E402

HOOKS = r"""
// A colour map (viridis, a polynomial fit).
float3 tack_viridis(float t)
{
    const float3 c0 = float3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    const float3 c1 = float3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    const float3 c2 = float3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    const float3 c3 = float3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    const float3 c4 = float3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    const float3 c5 = float3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    const float3 c6 = float3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

// A number of 24 bits as a colour of three bytes, exactly.
float4 tack_bytes(uint q)
{
    return float4(float((q >> 16) & 255u), float((q >> 8) & 255u), float(q & 255u), 255.f) / 255.f;
}

float4 tack_unit(float x)
{
    return tack_bytes(uint(clamp(x, 0.f, 1.f) * 16777215.f + 0.5f));
}

public struct TackSurface : IShaderHooks
{
    StructuredBuffer<uint> geometry;   // tack_geometry's arrays and scalars
    StructuredBuffer<uint> field;      // tack_value's
    float lo;                          // the colour map's range
    float hi;
    int per_face;                      // template triangles a face
    int place;                         // 1: the vertex hook places the points
    int probe;                         // 0: the picture; 1 value, 2 u, 3 v, 4 face + 1
    int base;                          // the group's first face in the surface
    int instanced;                     // 1: the face is the instance
    StructuredBuffer<float> placed;    // tack_place's positions and normals (computed)
    int computed;                      // 1: the vertex hook reads them from placed

    int face(uint cell, uint instance)
    {
        return instanced != 0 ? int(instance) : int(cell) / per_face;
    }

    public override void vertex(inout VertexContext v)
    {
        if (computed != 0) {
            uint at = 6u * (uint(face(v.cellId, v.instanceId)) * tack_template_points + v.pointId);
            v.positionMC = float4(placed[at], placed[at + 1u], placed[at + 2u], 1.f);
            v.normalMC = float3(placed[at + 3u], placed[at + 4u], placed[at + 5u]);
        } else if (place != 0) {
            float3x3 g = tack_geometry(geometry, face(v.cellId, v.instanceId), float3(v.uv, 0.f));
            v.positionMC = float4(g[0], 1.f);
            v.normalMC = normalize(cross(g[1], g[2]));
        }
    }

    public override void color(inout FragmentContext f)
    {
        float value = tack_value(field, face(f.cellId, f.instanceId), float3(f.uv, 0.f));
        f.diffuseColor = tack_viridis(saturate((value - lo) / (hi - lo)));
    }

    public override void output(inout FragmentContext f)
    {
        if (probe == 1)
            f.color = tack_unit((tack_value(field, face(f.cellId, f.instanceId), float3(f.uv, 0.f)) - lo) / (hi - lo));
        else if (probe == 2)
            f.color = tack_unit(f.uv.x);
        else if (probe == 3)
            f.color = tack_unit(f.uv.y);
        else if (probe == 4)
            f.color = tack_bytes(uint(base + face(f.cellId, f.instanceId) + 1));
    }
};
"""


def _words(array):
    out = vtkUnsignedIntArray()
    out.SetNumberOfValues(len(array))
    np.asarray(out)[:] = array
    return out


PLACE = r"""
// The placing pass: Tack's geometry at every face's template points, once per geometry
// change (vtkWebGPUShaderProperty.SetSlangComputedBuffer), for the vertex hook to read.
StructuredBuffer<uint> geometry;       // the property's "hooks.geometry"
RWStructuredBuffer<float> placed;      // the computed "hooks.placed": 6 floats a point

[shader("compute")]
[numthreads(64, 1, 1)]
void tack_place(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x + id.y * 4194240u;
    uint n, stride;
    placed.GetDimensions(n, stride);
    if (6u * i >= n)
        return;
    uint face = i / tack_template_points;
    uint k = i % tack_template_points;
    float3x3 g = tack_geometry(geometry, int(face), float3(tack_template[k], 0.f));
    float3 normal = normalize(cross(g[1], g[2]));
    for (uint c = 0u; c < 3u; ++c) {
        placed[6u * i + c] = g[0][c];
        placed[6u * i + 3u + c] = normal[c];
    }
}
"""


def _template_source(points):
    """The template's (u, v) as constants: the placing pass's points."""
    uv = ", ".join(f"float2({u!r}, {v!r})" for u, v in points.astype(np.float64).tolist())
    return (f"static const uint tack_template_points = {len(points)}u;\n"
            f"static const float2 tack_template[{len(points)}] = {{ {uv} }};\n")


def group_module(surface, name, group, points=None):
    """The group's Slang module (its type named after its source) and the two buffers'
    words: ``(source, type name, geometry words, field words)``. With the template's
    ``points``, the module has the placing pass too (``tack_place``)."""
    field = surface.fields[name]
    cells = surface.domain_view("cells", group)
    ids, pcs = _one_cell_arrays()
    geometry = export(_geometry_at, (cells, ids, pcs, tack.empty((3,), tack.vec3())),
                      "tack_geometry",
                      {"ids": ("int", 1), "pcs": ("float", 3), "out": ("float", 9)},
                      ("float3x3", "float3x3(" + ", ".join(f"{{out}}[{k}]" for k in range(9))
                       + ")"), binding=None)
    value = export(_value_at, (field.view(group), ids, pcs, tack.empty((1,), tack.f32)),
                   "tack_value", {"ids": ("int", 1), "pcs": ("float", 3), "out": ("float", 1)},
                   ("float", "{out}[0]"), binding=None)
    points = np.zeros((1, 2), np.float32) if points is None else points
    body = joined(geometry, value) + _template_source(points) + HOOKS + PLACE
    tag = hashlib.sha1(body.encode()).hexdigest()[:12]
    type_name = f"TackSurface_{tag}"
    source = (f"module tack_{tag};\nimport slangkit_hooks;\n\n"
              + body.replace("TackSurface", type_name))
    return source, type_name, geometry.buffers[0][1], value.buffers[0][1]


def _evaluate(cells, count, points):
    """Tack's geometry at each face's template points: positions and normals."""
    m = len(points)
    ids = tack.array(np.repeat(np.arange(count, dtype=np.int32), m))
    pcs = np.zeros((count * m, 3), np.float32)
    pcs[:, :2] = np.tile(points, (count, 1))
    geo = tack.empty((3 * count * m,), tack.vec3())
    _geometry_at(cells, ids, tack.array(pcs, tack.vec3()), geo)
    g = geo.numpy().reshape(-1, 3, 3)
    n = np.cross(g[:, 1], g[:, 2])
    return g[:, 0], n / np.linalg.norm(n, axis=1, keepdims=True)


def _corners_only(x, points, count, quad):
    """``x`` replaced by the face's corners interpolated (bilinearly, or linearly on a
    triangle): straight faces, for the vertex hook to bend."""
    m = len(points)
    x = x.reshape(count, m, 3)
    u, v = points[:, 0][None, :, None], points[:, 1][None, :, None]

    def at(pu, pv):
        return x[:, int(np.flatnonzero((points[:, 0] == pu) & (points[:, 1] == pv))[0])][:, None]

    if quad:
        out = ((1 - u) * (1 - v) * at(0, 0) + u * (1 - v) * at(1, 0) + (1 - u) * v * at(0, 1)
               + u * v * at(1, 1))
    else:
        out = (1 - u - v) * at(0, 0) + u * at(1, 0) + v * at(0, 1)
    return out.reshape(-1, 3)


class GroupActor:
    """One launch group as a VTK actor drawn through the hooks."""

    def __init__(self, surface, name, group, level, place, lo, hi, base, instanced=False):
        self.name, self.level, self.instanced = name, level, instanced
        self.place = place or instanced
        # Instanced: the placing pass computes the points once per geometry change
        # (COMPUTED=0: the vertex hook evaluates them every frame).
        self.computed = instanced and os.environ.get("COMPUTED", "1") != "0"
        self.source = None
        self.geometry_words = self.field_words = None
        self.mapper = rr.vtkPolyDataMapper()
        self.mapper.ScalarVisibilityOff()
        self.actor = rr.vtkActor(mapper=self.mapper)
        self.uniforms = self.actor.GetShaderProperty().GetFragmentCustomUniforms()
        self.uniforms.SetUniformf("hooks.lo", lo)
        self.uniforms.SetUniformf("hooks.hi", hi)
        self.uniforms.SetUniformi("hooks.place", int(self.place))
        self.uniforms.SetUniformi("hooks.instanced", int(instanced))
        self.uniforms.SetUniformi("hooks.computed", 0)
        self.uniforms.SetUniformi("hooks.probe", 0)
        self.set_surface(surface, group, base)

    def set_surface(self, surface, group, base):
        """``group`` of a new ``surface`` (new geometry, topology and field; the same
        shape and orders): Tack's module (recompiled by VTK only if its source changed),
        its buffers (refilled in place if the same size), a new polydata. Times Tack's
        parts in ``self.timings`` (s)."""
        import time

        t = time.perf_counter()
        quad = group.shape.NUM_POINTS == 4
        curved = surface.geometry.space.order > 1
        self.group, self.base, self.count = group, base, group.count
        points, tris = indexed_template(quad, self.level if curved else 1)
        source, type_name, gwords, fwords = group_module(surface, self.name, group, points)
        timings = {"export": time.perf_counter() - t}
        t = time.perf_counter()
        self.points = points
        m, per_face = len(points), len(tris) // 3
        if self.instanced:
            self._set_instanced(surface, group, quad, curved, points, tris, timings, t)
            self._set_module(source, type_name, gwords, fwords, per_face, base)
            timings["polydata"] = time.perf_counter() - t - timings["evaluate"]
            self.timings = timings
            return
        x, normals = _evaluate(surface.domain_view("cells", group), group.count, points)
        if self.place:
            x = _corners_only(x, points, group.count, quad)
        timings["evaluate"] = time.perf_counter() - t
        t = time.perf_counter()
        conn = (tris[None, :] + m * np.arange(group.count)[:, None]).reshape(-1).astype(np.int64)
        poly = vtkPolyData()
        poly.points = vtkPoints(data=vtkFloatArray(x.astype(np.float32)))
        poly.polys = vtkCellArray(offsets=np.arange(0, len(conn) + 1, 3), connectivity=conn)
        uv = vtkFloatArray(np.tile(points, (group.count, 1)).astype(np.float32))
        uv.SetName("uv")
        poly.GetPointData().SetTCoords(uv)
        if not self.place:
            nv = vtkFloatArray(normals.astype(np.float32))
            nv.SetName("Normals")
            poly.GetPointData().SetNormals(nv)
        self.poly, self.triangles = poly, len(conn) // 3
        self.mapper.SetInputData(poly)
        self._set_module(source, type_name, gwords, fwords, per_face, base)
        timings["polydata"] = time.perf_counter() - t
        self.timings = timings

    def _set_instanced(self, surface, group, quad, curved, points, tris, timings, t):
        """One face's template as the polydata, drawn once per face; the bounds from
        Tack's geometry at a coarse template (level 2), padded by 2% of the diagonal for
        the curvature between its points."""
        import time

        coarse = indexed_template(quad, 2 if curved else 1)[0]
        x, _ = _evaluate(surface.domain_view("cells", group), group.count, coarse)
        lo, hi = x.min(axis=0), x.max(axis=0)
        pad = 0.02 * float(np.linalg.norm(hi - lo))
        timings["evaluate"] = time.perf_counter() - t
        if getattr(self, "poly", None) is None or self.poly.GetNumberOfPoints() != len(points):
            # The template, kept across steps: (u, v, 0) points, (u, v) coordinates.
            poly = vtkPolyData()
            xyz = np.zeros((len(points), 3), np.float32)
            xyz[:, :2] = points
            poly.points = vtkPoints(data=vtkFloatArray(xyz))
            conn = tris.astype(np.int64)
            poly.polys = vtkCellArray(offsets=np.arange(0, len(conn) + 1, 3), connectivity=conn)
            uv = vtkFloatArray(points.astype(np.float32))
            uv.SetName("uv")
            poly.GetPointData().SetTCoords(uv)
            self.poly = poly
            self.mapper.SetInputData(poly)
        self.triangles = group.count * (len(tris) // 3)
        sp = self.actor.GetShaderProperty()
        if self.computed:
            # Tack's geometry at the template points, once per geometry change, on the GPU.
            sp.SetSlangComputedBuffer("hooks.placed", "tack_place", 6 * group.count * len(points),
                                      group.count * len(points))
        self.uniforms.SetUniformi("hooks.computed", int(self.computed))
        self.mapper.SetNumberOfInstances(group.count)
        self.mapper.SetDrawnBounds(lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad,
                                   lo[2] - pad, hi[2] + pad)

    def _set_module(self, source, type_name, gwords, fwords, per_face, base):
        sp = self.actor.GetShaderProperty()
        if not self.computed and sp.GetSlangBuffer("hooks.placed") is None:
            # The hook declares the placing pass's buffer: a word to bind when unused.
            self.unused_placed = _words(np.zeros(1, np.uint32))
            sp.SetSlangBuffer("hooks.placed", self.unused_placed)
        if source != self.source:
            sp.SetSlangModule(source)
            sp.SetSlangSpecialization("IShaderHooks", type_name)
            self.source, self.type_name = source, type_name
        for attr, words, path in (("geometry_words", gwords, "hooks.geometry"),
                                  ("field_words", fwords, "hooks.field")):
            held = getattr(self, attr)
            if held is not None and held.GetNumberOfValues() == len(words):
                np.asarray(held)[:] = words
                held.Modified()
            else:
                setattr(self, attr, _words(words))
                sp.SetSlangBuffer(path, getattr(self, attr))
        self.uniforms.SetUniformi("hooks.per_face", per_face)
        self.uniforms.SetUniformi("hooks.base", base)

    def set_probe(self, probe):
        self.uniforms.SetUniformi("hooks.probe", probe)

    def update_field(self, surface, name):
        """A new field of the same space: its words into the field buffer (the layout is
        the space's, so only the values move); VTK uploads the array, no new binding."""
        fwords = group_module(surface, name, self.group)[3]
        np.asarray(self.field_words)[:] = fwords
        self.field_words.Modified()


def surface_actors(surface, name, level=8, place=False, margin=0.05, instanced=None):
    """Every launch group of ``surface``'s field ``name`` as a ``GroupActor``, and the
    colour range ``(lo, hi)``. ``instanced`` None: the environment's ``INSTANCED=1``
    (so every script here runs either way)."""
    if instanced is None:
        instanced = os.environ.get("INSTANCED") == "1"
    values = surface.fields[name].values.numpy().reshape(-1)
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo or 1.0
    lo, hi = lo - margin * span, hi + margin * span
    actors, base = [], 0
    for group in surface.launch_groups("cells", [surface.fields[name]]):
        if not group.count:
            continue
        actors.append(GroupActor(surface, name, group, level, place, lo, hi, base, instanced))
        base += group.count
    return actors, (lo, hi)


def update_surface(actors, surface, name):
    """A new surface into the actors (its launch groups in the same order): Tack's
    times summed over the groups (s)."""
    groups = [g for g in surface.launch_groups("cells", [surface.fields[name]]) if g.count]
    if len(groups) != len(actors):
        raise ValueError("the new surface has other launch groups")
    base, total = 0, {}
    for a, g in zip(actors, groups, strict=True):
        a.set_surface(surface, g, base)
        base += g.count
        for k, v in a.timings.items():
            total[k] = total.get(k, 0.0) + v
    return total
