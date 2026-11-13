# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Surface LIC drawn by VTK itself (the WebGPU renderer on the Slang branch): a slice of
``disk_out_ref`` as a polydata through VTK's mapper, Tack's fields evaluated per pixel
by a Slang hook, and ``vtkWebGPUSurfaceLIC`` on the renderer convolving what the hook
wrote. The same scene the browser will get through VTK's serialization.

- **The hook** (``output``): Newton's method in the pixel's hexahedron (the triangle's
  cell, from ξ = 0.5, on Tack's ``tack_cell_geometry``), ``V`` and ``Temp`` there, the
  colour (``Temp`` through viridis, a headlight), and into the ids target the field's
  screen direction (two halves), its tangential magnitude, the view depth and a noise
  fixed to the surface (see ``vtkWebGPUSurfaceLIC.h``).
- **The data:** one buffer (``exact.volume_functions`` over the cut cells) and each
  triangle's cell, as ``SetSlangBuffer`` arrays.

    python vtk_lic.py [OUTDIR]
"""

import hashlib
import os
import sys
import time

import numpy as np

import vtkmodules.vtkCommonDataModel  # noqa: F401
import vtkmodules.vtkRenderingCore as rr
import vtkmodules.vtkRenderingWebGPU  # noqa: F401
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkPoints, vtkUnsignedIntArray
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkRenderingWebGPU import vtkWebGPUShaderProperty, vtkWebGPUSurfaceLIC

import disk
import exact
import scene
import tack
import tack.dataset as td

HOOK = r"""
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

float3x3 tack_inverse3(float3x3 m)
{
    float3 a = m[0], b = m[1], c = m[2];
    float3 r0 = cross(b, c), r1 = cross(c, a), r2 = cross(a, b);
    return transpose(float3x3(r0, r1, r2)) / dot(a, r0);
}

// The point x's parametric coordinates in cell c: Newton's method on Tack's geometry.
float3 tack_locate(StructuredBuffer<uint> pool, int c, float3 x, float3 xi)
{
    for (int it = 0; it < 8; ++it) {
        float4x3 g = tack_cell_geometry(pool, c, xi);
        float3x3 jacobian = transpose(float3x3(g[1], g[2], g[3]));
        float3 step = mul(tack_inverse3(jacobian), x - g[0]);
        xi += step;
        if (dot(step, step) < 1e-14f)
            break;
    }
    return xi;
}

// Pixels moved per unit of v at p (the pixel's y down).
float2 tack_screen(float4x4 m, float3 p, float3 v, float2 viewport)
{
    float4 c = mul(m, float4(p, 1.f));
    float4 dc = mul(m, float4(v, 0.f));
    float2 dndc = (dc.xy * c.w - c.xy * dc.w) / (c.w * c.w);
    return float2(dndc.x * 0.5f * viewport.x, -dndc.y * 0.5f * viewport.y);
}

float tack_lattice_noise(int3 cell, uint octave)
{
    uint3 v = uint3(cell) * 1664525u + 1013904223u + octave * 2654435769u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return float(v.x & 0xffffffu) / 16777215.f;
}

public struct TackLIC : IShaderHooks
{
    StructuredBuffer<uint> pool;       // Tack's arrays (the cut cells)
    StructuredBuffer<uint> cells;      // each triangle's cell
    float lo;                          // Temp's colour range
    float hi;
    float noise_pixels;                // the noise's cell, in pixels
    float nx;                          // the plane's normal
    float ny;
    float nz;

    public override void output(inout FragmentContext f)
    {
        int c = int(cells[f.cellId]);
        float3 p = f.positionMC;
        float3 xi = tack_locate(pool, c, p, float3(0.5f));
        float3 v = tack_vector(pool, c, xi);
        float temp = tack_scalar(pool, c, xi);
        float3 n = normalize(float3(nx, ny, nz));
        float3 vt = v - dot(v, n) * n;

        float3 view = normalize(-f.positionVC.xyz);
        float light = 0.3f + 0.7f * abs(dot(normalize(f.normalVC), view));
        f.color = float4(tack_viridis(saturate((temp - lo) / (hi - lo))) * light, 1.f);

        float2 d = tack_screen(f.modelToClip, p, vt, f.viewport);
        float len = length(d);
        float2 dir = len > 0.f ? d / len : float2(0.f);
        float w = mul(f.modelToClip, float4(p, 1.f)).w;

        // Noise fixed to the surface: a lattice in model space whose cell is about
        // noise_pixels pixels (the larger of the plane's two pixel footprints), two
        // octaves faded.
        float3 t1 = normalize(abs(n.x) < 0.9f ? cross(n, float3(1.f, 0.f, 0.f))
                                              : cross(n, float3(0.f, 1.f, 0.f)));
        float3 t2 = cross(n, t1);
        float l1 = length(tack_screen(f.modelToClip, p, t1, f.viewport));
        float l2 = length(tack_screen(f.modelToClip, p, t2, f.viewport));
        float footprint = 1.f / max(min(l1, l2), 1e-9f);
        float level = log2(max(noise_pixels * footprint, 1e-9f));
        float l0 = floor(level);
        float fade = level - l0;
        uint o0 = uint(int(l0) + 64);
        float n0 = tack_lattice_noise(int3(floor(p / exp2(l0))), o0);
        float n1 = tack_lattice_noise(int3(floor(p / exp2(l0 + 1.f))), o0 + 1u);

        f.ids = uint4(f32tof16(dir.x) | (f32tof16(dir.y) << 16), asuint(length(vt)), asuint(w),
                      asuint(lerp(n0, n1, fade)));
    }
};
"""


def words(array, dtype=np.uint32):
    out = vtkUnsignedIntArray()
    out.SetNumberOfValues(len(array))
    np.asarray(out)[:] = np.asarray(array, dtype).view(np.uint32)
    return out


class SliceActor:
    """The slice as a VTK actor drawn through the hook, for any plane through ``data``."""

    def __init__(self, data, temp_range, noise_pixels=1.5):
        self.data = data
        self.locator = td.CellLocator(data)
        self.mapper = rr.vtkPolyDataMapper()
        self.mapper.ScalarVisibilityOff()
        self.property = vtkWebGPUShaderProperty()
        self.actor = rr.vtkActor(mapper=self.mapper)
        self.actor.SetShaderProperty(self.property)
        self.uniforms = self.property.GetFragmentCustomUniforms()
        self.uniforms.SetUniformf("hooks.lo", temp_range[0])
        self.uniforms.SetUniformf("hooks.hi", temp_range[1])
        self.uniforms.SetUniformf("hooks.noise_pixels", noise_pixels)
        self.source = None
        self.timings = {}

    def cut(self, origin, normal):
        """The slice through ``origin`` with ``normal``: Tack cuts and packs; the actor's
        polydata, buffers and plane are replaced (the Slang only when its structure is
        new: the same for every plane through one mesh)."""
        t0 = time.perf_counter()
        points, triangles, _, unit, _ = scene.slice_triangles(
            self.data, tuple(origin), tuple(normal), vector="V", scalar="Temp")
        k = len(triangles)
        t1 = time.perf_counter()
        poly = vtkPolyData()
        if k:
            cells, _, _, _ = exact.slice_cells(self.data, points, triangles, locator=self.locator)
            cut, cells = exact.cut_cells(self.data, cells)
            body, pool = exact.volume_functions(cut, "V", "Temp")
            body += HOOK
            if body != getattr(self, "body", None):
                tag = hashlib.sha1(body.encode()).hexdigest()[:12]
                self.property.SetSlangModule(f"module tack_lic_{tag};\nimport slangkit_hooks;\n\n"
                                             + body.replace("TackLIC", f"TackLIC_{tag}"))
                self.property.SetSlangSpecialization("IShaderHooks", f"TackLIC_{tag}")
                self.body = body
            self.pool, self.cells = words(pool), words(cells)
            self.property.SetSlangBuffer("hooks.pool", self.pool)
            self.property.SetSlangBuffer("hooks.cells", self.cells)
            poly.points = vtkPoints(data=vtkFloatArray(points.astype(np.float32)))
            conn = triangles.reshape(-1).astype(np.int64)
            poly.polys = vtkCellArray(offsets=np.arange(0, len(conn) + 1, 3), connectivity=conn)
        for name, value in zip("xyz", unit, strict=True):
            self.uniforms.SetUniformf(f"hooks.n{name}", float(value))
        self.mapper.SetInputData(poly)
        self.timings = {"slice": 1000 * (t1 - t0), "pack": 1000 * (time.perf_counter() - t1),
                        "triangles": k}


def main(out="."):
    tack.set_default_device("cpu")
    data, bounds = disk.read(disk.DISK)
    lo, hi = np.array(bounds[::2]), np.array(bounds[1::2])
    centre = 0.5 * (lo + hi)
    temp = data.fields["Temp"].values.numpy().reshape(-1)
    slice_actor = SliceActor(data, (float(temp.min()), float(temp.max())))
    slice_actor.cut(centre, (1.0, 0.0, 0.0))

    renderer = rr.vtkRenderer(background=(1.0, 1.0, 1.0))
    renderer.AddActor(slice_actor.actor)
    lic = vtkWebGPUSurfaceLIC()
    renderer.SetSurfaceLIC(lic)
    window = rr.vtkRenderWindow(size=(1920, 1080), off_screen_rendering=True)
    window.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    span = float(np.max(hi - lo))
    camera.SetPosition(*(centre + np.array([1.7 * span, 0.0, 0.0])))
    camera.SetFocalPoint(*centre)
    camera.SetViewUp(0.0, 1.0, 0.0)
    camera.SetViewAngle(30.0)
    renderer.ResetCameraClippingRange()
    t = time.perf_counter()
    window.Render()
    print(f"first frame {1000 * (time.perf_counter() - t):.0f} ms", flush=True)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        window.Render()
        window.WaitForCompletion()
        times.append(1000 * (time.perf_counter() - t))
    print(f"frames: median {np.median(times):.1f} ms", flush=True)
    picture = grab(window)
    import lic as wgpu_lic
    wgpu_lic.write_png(os.path.join(out, "vtk_lic.png"), picture)
    check(data, picture, camera, window.GetSize(), centre, np.array([1.0, 0.0, 0.0]))
    print(f"  picture in {os.path.abspath(out)}", flush=True)


def grab(window):
    from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter

    g = vtkWindowToImageFilter(input=window, input_buffer_type=3)
    g.Update()
    img = g.GetOutput()
    w, h, _ = img.GetDimensions()
    rgba = np.asarray(img.GetPointData().GetScalars()).reshape(h, w, -1)
    return np.ascontiguousarray(rgba[::-1])           # row 0 at the top


def check(data, picture, camera, size, origin, normal, radius=6):
    """The picture's streaks (its luminance's structure tensor) against V's screen
    direction computed independently: each pixel's ray onto the plane, V probed there,
    projected with the camera."""
    import lic as wgpu_lic
    from renderer import look_at, perspective

    w, h = size
    rows = perspective(camera.GetViewAngle(), w / h, *camera.GetClippingRange()) @ look_at(
        np.array(camera.GetPosition()), np.array(camera.GetFocalPoint()),
        np.array(camera.GetViewUp()))
    inverse = np.linalg.inv(rows)
    j, i = np.mgrid[0:h, 0:w]
    ndc = np.stack([(i + 0.5) / w * 2 - 1, 1 - (j + 0.5) / h * 2], axis=-1).reshape(-1, 2)

    def unproject(z):
        clip = np.concatenate([ndc, np.full((len(ndc), 1), z), np.ones((len(ndc), 1))], axis=1)
        p = clip @ inverse.T
        return p[:, :3] / p[:, 3:]

    near, far = unproject(0.0), unproject(1.0)
    d = far - near
    where = near + (((origin - near) @ normal) / (d @ normal))[:, None] * d
    out = td.probe(data, where, fields=["V"])
    valid = out.fields["valid"].values.numpy().reshape(-1) > 0
    v = out.fields["V"].values.numpy().reshape(-1, 3)
    vt = v - (v @ normal)[:, None] * normal
    c = np.concatenate([where, np.ones((len(where), 1))], axis=1) @ rows.T
    dc = np.concatenate([vt, np.zeros((len(vt), 1))], axis=1) @ rows.T
    dndc = (dc[:, :2] * c[:, 3:] - c[:, :2] * dc[:, 3:]) / c[:, 3:] ** 2
    field = np.zeros((h * w, 4), np.float32)
    field[:, 0], field[:, 1] = dndc[:, 0] * w, -dndc[:, 1] * h
    field[:, :2] /= np.maximum(np.linalg.norm(field[:, :2], axis=1, keepdims=True), 1e-30)
    field[:, 2] = np.where(valid, np.linalg.norm(vt, axis=1), -1.0)
    luminance = picture[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    err = wgpu_lic.orientation_error(luminance / 255.0, field.reshape(h, w, 4), radius)
    print(f"  orientation vs V's screen direction over {err.size} pixels: median "
          f"{np.median(err):.1f} deg, 90th percentile {np.percentile(err, 90):.1f} deg",
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
