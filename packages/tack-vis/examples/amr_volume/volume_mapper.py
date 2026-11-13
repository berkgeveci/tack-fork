# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``VolumeMapper``: a WebGPU mapper, written in Python, that volume renders a
``TackVolume`` output by marching rays through it in a Slang hook. Users make one as
any mapper::

    mapper = VolumeMapper()
    mapper.SetInputConnection(tack_volume.GetOutputPort())
    mapper.color.AddRGBPoint(...)          # a vtkColorTransferFunction
    mapper.opacity.AddPoint(...)           # a vtkPiecewiseFunction
    actor = vtkActor(mapper=mapper)
    actor.ForceTranslucentOn()             # composited through the renderer's A-buffer

The input is the domain's bounding box (triangles) carrying what the field is: the
pool (``TackPool``), ``tack_sample(pool, p)`` as Slang (``TackEvaluation``), the
bounds, the value range and the cell size. ``RequestSlang`` builds the module from
that evaluation and the marcher below, binds the pool, and samples the two transfer
functions over the value range into a table. Nothing in it depends on the camera: the
hook finds each pixel's ray from per-frame state, so a browser's VTK (no Python)
turning the camera draws correctly from the state it was sent.

**The marcher** (``HOOK``): fragments of the box's front faces are discarded (``clip``),
so each pixel's fragment is on a back face, its ``positionMC`` the ray's far end; the
inverse of ``modelToClip`` unprojects the same pixel onto the near plane, the ray's
start. A slab test against the bounds gives where it enters. From there:

- in a macrocell the transfer function makes invisible (``hooks.visible``, 0), the ray
  jumps to where it leaves it;
- otherwise a step ``h`` is ``sample_distance`` times the local cell size
  (``tack_cell_size``: the finest level holding the point), sampled at its middle;
  the transfer function by linear interpolation in its table; opacity corrected for
  the step (``1 - (1 - a)^(h / unit)``); composited front to back until the opacity
  reaches ``volume_tack.OPAQUE``.

The hook writes ``(C / A, A)``, which the A-buffer composites over the opaque image.
``volume_tack.reference`` is the same march in Tack, and ``visible`` there what makes
``hooks.visible`` (a Tack kernel over the input's macrocell ranges and the table).
"""

import hashlib

import numpy as np
import volume_tack
from vtkmodules.util.vtkMapper import VTKPythonMapperBase
from vtkmodules.vtkCommonCore import vtkUnsignedIntArray
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction
from vtkmodules.vtkRenderingWebGPU import vtkWebGPUShaderProperty

HOOK = r"""
// The inverse of a 4x4 matrix (by 2x2 sub-determinants; the layout's convention
// does not matter, since inversion commutes with transposition).
float4x4 tack_inverse4(float4x4 m)
{
    float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], a03 = m[0][3];
    float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], a13 = m[1][3];
    float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], a23 = m[2][3];
    float a30 = m[3][0], a31 = m[3][1], a32 = m[3][2], a33 = m[3][3];
    float b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
    float b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
    float b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
    float b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
    float b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
    float b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    float det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    float4x4 r = float4x4(
        a11 * b11 - a12 * b10 + a13 * b09, a02 * b10 - a01 * b11 - a03 * b09,
        a31 * b05 - a32 * b04 + a33 * b03, a22 * b04 - a21 * b05 - a23 * b03,
        a12 * b08 - a10 * b11 - a13 * b07, a00 * b11 - a02 * b08 + a03 * b07,
        a32 * b02 - a30 * b05 - a33 * b01, a20 * b05 - a22 * b02 + a23 * b01,
        a10 * b10 - a11 * b08 + a13 * b06, a01 * b08 - a00 * b10 - a03 * b06,
        a30 * b04 - a31 * b02 + a33 * b00, a21 * b02 - a20 * b04 - a23 * b00,
        a11 * b07 - a10 * b09 - a12 * b06, a00 * b09 - a01 * b07 + a02 * b06,
        a31 * b01 - a30 * b03 - a32 * b00, a20 * b03 - a21 * b01 + a22 * b00);
    return r * (1.f / det);
}

// The transfer function at t in [0, 1]: linear between its n entries (RGBA, as words).
float4 tack_transfer(StructuredBuffer<uint> tf, int n, float t)
{
    float x = min(max(t, 0.f), 1.f) * float(n - 1);
    int i = min(int(x), n - 2);
    float f = x - float(i);
    float4 a = float4(asfloat(tf[4 * i]), asfloat(tf[4 * i + 1]), asfloat(tf[4 * i + 2]),
                      asfloat(tf[4 * i + 3]));
    float4 b = float4(asfloat(tf[4 * i + 4]), asfloat(tf[4 * i + 5]), asfloat(tf[4 * i + 6]),
                      asfloat(tf[4 * i + 7]));
    return a * (1.f - f) + b * f;
}

float tack_safe_inverse(float x)
{
    return 1.f / (abs(x) < 1e-20f ? (x < 0.f ? -1e-20f : 1e-20f) : x);
}

float tack_exit(float q, float d, float inv, int mi, float m)
{
    float face = float(mi) * m;
    if (d >= 0.f)
        face = float(mi + 1) * m;
    return (face - q) * inv;
}

public struct TackVolumeHook : IShaderHooks
{
    StructuredBuffer<uint> pool;       // Tack's arrays
    StructuredBuffer<uint> tf;         // the transfer function's table
    StructuredBuffer<uint> visible;    // per macrocell: can it show (1) or not (0)
    float tf_size;                     // the table's entries
    float lo;                          // the value range the table spans
    float hi;
    float x0;                          // the bounds
    float y0;
    float z0;
    float x1;
    float y1;
    float z1;
    float factor;                      // a step, in local cells
    float unit;                        // the opacity's unit distance
    float skip;                        // past a macrocell's face
    float gx;                          // the macrocells: how many,
    float gy;
    float gz;
    float mox;                         // where they start,
    float moy;
    float moz;
    float mx;                          // and their size
    float my;
    float mz;
    float max_steps;
    float opaque;

    // One fragment a pixel: the back face's.
    public override bool clip(inout FragmentContext f) { return !f.frontFacing; }

    public override void output(inout FragmentContext f)
    {
        float3 back = f.positionMC;
        float4 c = mul(f.modelToClip, float4(back, 1.f));
        float4 q = mul(tack_inverse4(f.modelToClip), float4(c.xy / c.w, 0.f, 1.f));
        float3 near = q.xyz / q.w;
        float3 d = back - near;
        float3 dir = d / length(d);
        float3 inv = float3(tack_safe_inverse(dir.x), tack_safe_inverse(dir.y),
                            tack_safe_inverse(dir.z));
        float3 ta = (float3(x0, y0, z0) - near) * inv;
        float3 tb = (float3(x1, y1, z1) - near) * inv;
        float3 tn = min(ta, tb), tx = max(ta, tb);
        float t = max(max(max(tn.x, tn.y), tn.z), 0.f);
        float t1 = min(min(tx.x, tx.y), tx.z);
        float3 o = near - float3(mox, moy, moz);
        int ngx = int(gx), ngy = int(gy), ngz = int(gz);
        int most = int(max_steps);
        int entries = int(tf_size);
        float3 colour = float3(0.f);
        float a = 0.f;
        int steps = 0;
        while (t < t1 && steps < most) {
            steps += 1;
            int mi = min(max(int(floor((o.x + dir.x * t) / mx)), 0), ngx - 1);
            int mj = min(max(int(floor((o.y + dir.y * t) / my)), 0), ngy - 1);
            int mk = min(max(int(floor((o.z + dir.z * t) / mz)), 0), ngz - 1);
            if (visible[mi + ngx * (mj + ngy * mk)] == 0u) {
                float ex = tack_exit(o.x, dir.x, inv.x, mi, mx);
                float ey = tack_exit(o.y, dir.y, inv.y, mj, my);
                float ez = tack_exit(o.z, dir.z, inv.z, mk, mz);
                t = max(min(min(ex, ey), ez), t) + skip;
            } else {
                float h = factor * tack_cell_size(pool, near + dir * t);
                float v = tack_sample(pool, near + dir * (t + 0.5f * h));
                float4 s = tack_transfer(tf, entries, (v - lo) / (hi - lo));
                float alpha = 1.f - pow(1.f - s.w, h / unit);
                float w = (1.f - a) * alpha;
                colour += w * s.xyz;
                a += w;
                t += h;
                if (a >= opaque)
                    break;
            }
        }
        f.color = a > 0.f ? float4(colour / a, a) : float4(0.f);
    }
};
"""


def _words(values):
    out = vtkUnsignedIntArray()
    out.SetNumberOfValues(len(values))
    np.asarray(out)[:] = np.asarray(values).view(np.uint32)
    return out


class VolumeMapper(VTKPythonMapperBase):
    """Volume renders a ``TackVolume`` output (see the module's docstring). ``color``
    and ``opacity`` are the transfer functions, in data values; ``sample_distance`` is
    the step in local cells, ``unit_distance`` the opacity's unit in the finest cells;
    ``table_size`` the transfer function table's entries."""

    def __init__(self, sample_distance=0.5, unit_distance=1.0, table_size=256):
        VTKPythonMapperBase.__init__(self)
        self.ScalarVisibilityOff()
        self.slang = vtkWebGPUShaderProperty()
        self.SetSlangShaderProperty(self.slang)
        self.uniforms = self.slang.GetFragmentCustomUniforms()
        self.color = vtkColorTransferFunction()
        self.opacity = vtkPiecewiseFunction()
        self.sample_distance = sample_distance
        self.skip_empty = True                # empty-space skipping (off: for measuring)
        self.unit_distance = unit_distance
        self.table_size = table_size
        self.body = None
        self.seen = None
        self.preparations = 0

    def table(self, lo, hi):
        """The transfer functions sampled over ``[lo, hi]``: ``(table_size, 4)`` f32."""
        x = np.linspace(lo, hi, self.table_size)
        rgb = np.array([self.color.GetColor(v) for v in x])
        a = np.array([self.opacity.GetValue(v) for v in x])
        return np.concatenate([rgb, a[:, None]], axis=1).astype(np.float32)

    def march(self, poly):
        """What the hook marches with, from the input ``poly``: the table, its range,
        the step factor, the opacity's unit, the macrocells and which can show (a Tack
        kernel), the bounds and the skip past a macrocell's face."""
        import tack

        fields = poly.GetFieldData()
        lo, hi = fields.GetArray("TackRange").GetTuple2(0)
        cell = fields.GetArray("TackCellSize").GetValue(0)
        g = fields.GetArray("TackMacroGrid").GetTuple(0)
        grid = {"origin": g[0:3], "size": g[3:6], "dims": tuple(int(v) for v in g[6:9])}
        table = self.table(lo, hi)
        lows = tack.array(np.asarray(fields.GetArray("TackMacroLow")))
        highs = tack.array(np.asarray(fields.GetArray("TackMacroHigh")))
        if self.skip_empty:
            seen = volume_tack.visible(lows, highs, table, lo, hi)
        else:
            seen = tack.array(np.ones(len(lows), np.int32))
        return {"table": table, "lo": lo, "hi": hi, "factor": self.sample_distance,
                "unit": self.unit_distance * cell, "skip": 0.01 * cell, "grid": grid,
                "seen": seen,
                "bounds": fields.GetArray("TackBounds").GetTuple(0)}

    def RequestSlang(self, renderer, actor):
        poly = self.GetInput()
        if poly is None:
            return
        key = (poly.GetMTime(), self.color.GetMTime(), self.opacity.GetMTime(),
               self.sample_distance, self.unit_distance, self.table_size, self.skip_empty)
        if key == self.seen:
            return
        self.seen = key
        fields = poly.GetFieldData()
        evaluation = fields.GetAbstractArray("TackEvaluation")
        pool = fields.GetArray("TackPool")
        if evaluation is None or pool is None:
            return
        body = evaluation.GetValue(0) + HOOK
        if body != self.body:
            tag = hashlib.sha1(body.encode()).hexdigest()[:12]
            self.slang.SetSlangModule(f"module tack_volume_{tag};\nimport slangkit_hooks;\n\n"
                                      + body.replace("TackVolumeHook", f"TackVolumeHook_{tag}"))
            self.slang.SetSlangSpecialization("IShaderHooks", f"TackVolumeHook_{tag}")
            self.body = body
        march = self.march(poly)
        self.tf = _words(march["table"].reshape(-1))
        self.visible = _words(march["seen"].numpy().astype(np.uint32))
        self.slang.SetSlangBuffer("hooks.pool", pool)
        self.slang.SetSlangBuffer("hooks.tf", self.tf)
        self.slang.SetSlangBuffer("hooks.visible", self.visible)
        grid = march["grid"]
        values = {"tf_size": self.table_size, "lo": march["lo"], "hi": march["hi"],
                  "factor": march["factor"], "unit": march["unit"], "skip": march["skip"],
                  "max_steps": volume_tack.MAX_STEPS, "opaque": volume_tack.OPAQUE}
        values.update(zip(("x0", "x1", "y0", "y1", "z0", "z1"), march["bounds"], strict=True))
        values.update(zip(("gx", "gy", "gz"), grid["dims"], strict=True))
        values.update(zip(("mox", "moy", "moz"), grid["origin"], strict=True))
        values.update(zip(("mx", "my", "mz"), grid["size"], strict=True))
        for name, value in values.items():
            self.uniforms.SetUniformf(f"hooks.{name}", float(value))
        self.preparations += 1
