# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Image-space surface LIC with WebGPU (wgpu-py), every pass in Slang (``lic.slang``,
compiled to WGSL by ``slangc``): a planar slice of a linear mesh with a linear vector
field (``scene``), drawn as

    surface pass -> LIC -> contrast and high-pass -> LIC -> contrast -> composite

at the picture's resolution (``SUPERSAMPLE=2`` for twice), and checked: the LIC image's local orientation (the
structure tensor's) against the field's screen-space direction, with the raw noise as
the control.

    SLANGC=.../slangc PYTHONPATH=<wgpu-py> python lic.py [OUTDIR]
"""

import os
import re
import subprocess
import sys
import time

import numpy as np
import wgpu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "webgpu_surfaces"))

import scene  # noqa: E402
from renderer import look_at, perspective  # noqa: E402

import tack  # noqa: E402

SLANGC = os.environ.get("SLANGC", "slangc")
PASSES = {
    "surface": [("vs_surface", "vertex"), ("fs_surface", "fragment")],
    "lic": [("lic", "compute")],
    "histogram": [("histogram", "compute")],
    "percentiles": [("percentiles", "compute")],
    "enhance": [("enhance", "compute")],
    "composite": [("composite", "compute")],
}
# Bindings, by the numbers in lic.slang.
FRAME, FIELD, INPUT, OUTPUT, RANGE, COLOR, FINAL, POINTS, VECTORS, TRIANGLES, SCALARS = (
    0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13)
GEOMETRY_POOL, VECTOR_POOL, SCALAR_POOL, TRIANGLE_CELLS, PARAMETRIC = 14, 15, 16, 17, 18


def compile_passes(out_dir, defines=()):
    """Each pass's WGSL and the bindings it uses; ``defines`` for the surface pass
    (``TACK_EXACT`` includes ``out_dir``'s ``tack_volume.slang``)."""
    os.makedirs(out_dir, exist_ok=True)
    source = os.path.join(HERE, "lic.slang")
    modules = {}
    for name, entries in PASSES.items():
        out = os.path.join(out_dir, name + ".wgsl")
        args = [SLANGC, source, "-target", "wgsl", "-o", out, "-I", out_dir]
        if name == "surface":
            for d in defines:
                args += ["-D", d]
        for entry, stage in entries:
            args += ["-entry", entry, "-stage", stage]
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"slangc {name}:\n{r.stderr[:3000]}")
        with open(out) as f:
            code = f.read()
        used = sorted({int(b) for b in re.findall(r"@binding\((\d+)\) @group\(0\)", code)})
        modules[name] = (code, used)
    return modules


class LIC:
    def __init__(self, points, triangles, vectors, out_dir, size=(1280, 800), supersample=2,
                 scalars=None, exact=None):
        """``scalars`` (one a point), if given, colour the surface instead of the
        field's tangential speed (``render(..., by_scalar=True)``).

        ``exact`` evaluates the fields per pixel with the cells' own basis instead of
        interpolating ``vectors`` and ``scalars`` (then unused): a dict of ``source``
        (``exact.volume_functions``' Slang), ``pools`` (its three buffers' words),
        ``cells`` (a triangle's cell) and ``pcs`` (``(k, 3, 3)``, each triangle vertex's
        parametric coordinates there); ``points`` are then per triangle vertex
        (``(3k, 3)``) and ``triangles`` ``arange(3k)``."""
        self.size, self.ss = size, supersample
        self.w, self.h = size[0] * supersample, size[1] * supersample
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = d = adapter.request_device_sync()
        storage = wgpu.BufferUsage.STORAGE
        self.points = d.create_buffer_with_data(data=points.reshape(-1), usage=storage)
        self.vectors = d.create_buffer_with_data(data=vectors.reshape(-1), usage=storage)
        self.triangles = d.create_buffer_with_data(data=triangles.reshape(-1), usage=storage)
        scalars = np.zeros(len(points), np.float32) if scalars is None else scalars
        self.scalars = d.create_buffer_with_data(data=np.asarray(scalars, np.float32),
                                                 usage=storage)
        self.count = triangles.size
        self.frame = d.create_buffer(size=128, usage=storage | wgpu.BufferUsage.COPY_DST)
        self.ranges = [d.create_buffer(size=4 * 258, usage=storage | wgpu.BufferUsage.COPY_DST
                                       | wgpu.BufferUsage.COPY_SRC) for _ in range(2)]

        defines = ()
        self.exact = {}
        if exact is not None:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "tack_volume.slang"), "w") as f:
                f.write(exact["source"])
            defines = ("TACK_EXACT",)
            for b, words in zip((GEOMETRY_POOL, VECTOR_POOL, SCALAR_POOL), exact["pools"],
                                strict=True):
                self.exact[b] = d.create_buffer_with_data(
                    data=np.ascontiguousarray(words, np.uint32), usage=storage)
            self.exact[TRIANGLE_CELLS] = d.create_buffer_with_data(
                data=np.ascontiguousarray(exact["cells"], np.uint32), usage=storage)
            self.exact[PARAMETRIC] = d.create_buffer_with_data(
                data=np.ascontiguousarray(exact["pcs"], np.float32).reshape(-1), usage=storage)
        self.modules = compile_passes(out_dir, defines)
        code, self.surface_bindings = self.modules["surface"]
        module = d.create_shader_module(code=code)
        self.surface = d.create_render_pipeline(
            layout="auto",
            vertex={"module": module, "entry_point": "vs_surface", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list,
                       "cull_mode": wgpu.CullMode.none},
            depth_stencil={"format": wgpu.TextureFormat.depth32float,
                           "depth_write_enabled": True,
                           "depth_compare": wgpu.CompareFunction.less},
            fragment={"module": module, "entry_point": "fs_surface",
                      "targets": [{"format": wgpu.TextureFormat.rgba32float},
                                  {"format": wgpu.TextureFormat.rgba8unorm},
                                  {"format": wgpu.TextureFormat.r32float}]})
        self.compute = {}
        for name in ("lic", "histogram", "percentiles", "enhance", "composite"):
            code, used = self.modules[name]
            pipeline = d.create_compute_pipeline(
                layout="auto", compute={"module": d.create_shader_module(code=code),
                                        "entry_point": name})
            self.compute[name] = (pipeline, used)

        T = wgpu.TextureUsage
        w, h = self.w, self.h

        def texture(fmt, usage):
            return d.create_texture(size=(w, h, 1), format=fmt, usage=usage)

        self.field = texture(wgpu.TextureFormat.rgba32float,
                             T.RENDER_ATTACHMENT | T.TEXTURE_BINDING | T.COPY_SRC)
        self.color = texture(wgpu.TextureFormat.rgba8unorm,
                             T.RENDER_ATTACHMENT | T.TEXTURE_BINDING | T.COPY_SRC)
        self.noise = texture(wgpu.TextureFormat.r32float,
                             T.RENDER_ATTACHMENT | T.TEXTURE_BINDING | T.COPY_SRC)
        self.depth = texture(wgpu.TextureFormat.depth32float, T.RENDER_ATTACHMENT)
        self.a = texture(wgpu.TextureFormat.r32float,
                         T.STORAGE_BINDING | T.TEXTURE_BINDING | T.COPY_SRC)
        self.b = texture(wgpu.TextureFormat.r32float,
                         T.STORAGE_BINDING | T.TEXTURE_BINDING | T.COPY_SRC)
        self.final = texture(wgpu.TextureFormat.rgba8unorm, T.STORAGE_BINDING | T.COPY_SRC)

    def _bind(self, pipeline, used, resources):
        return self.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{"binding": b, "resource": resources[b]} for b in used])

    def _dispatch(self, encoder, name, resources, one=False):
        pipeline, used = self.compute[name]
        p = encoder.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, self._bind(pipeline, used, resources))
        p.dispatch_workgroups(*((1, 1, 1) if one else ((self.w + 7) // 8, (self.h + 7) // 8, 1)))
        p.end()

    def render(self, eye, target, up, field_range, noise_pixels=1.5, step=1.0, steps=20,
               depth_jump=0.05, smallest=1e-3, highpass=0.6, clip=0.01, enhanced=True,
               fovy=30.0, by_scalar=False, check=False):
        """The picture (``(h, w, 4)`` u8, downsampled) and, at full resolution, the
        field target, the noise and the LIC intensity; GPU times (ms). With ``check``,
        the surface pass alone, its field target holding the field and the scalar
        themselves per pixel (``values``, ``(h, w, 4)``: x, y, z, scalar; scalar 0
        where no surface), and ``rows``, the projection the pixels are of."""
        d, w, h = self.device, self.w, self.h
        eye, target = np.asarray(eye, float), np.asarray(target, float)
        distance = float(np.linalg.norm(eye - target))
        rows = perspective(fovy, w / h, 0.05 * distance, 4.0 * distance) @ look_at(
            eye, target, np.asarray(up, float))
        frame = np.zeros((8, 4), np.float32)
        frame[:4] = rows
        frame[4] = (*eye, float(check))
        frame[5] = (field_range[0], field_range[1], noise_pixels, float(by_scalar))
        frame[6] = (w, h, step * self.ss, steps * self.ss)
        frame[7] = (depth_jump, smallest, highpass, clip)
        d.queue.write_buffer(self.frame, 0, frame)
        start = np.zeros(258, np.uint32)
        for r in self.ranges:
            d.queue.write_buffer(r, 0, start)

        # The surface pass.
        timings = {}
        t0 = time.perf_counter()
        encoder = d.create_command_encoder()
        rp = encoder.begin_render_pass(
            color_attachments=[
                {"view": self.field.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, -1, 0)},
                {"view": self.color.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store, "clear_value": (1, 1, 1, 1)},
                {"view": self.noise.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 0)}],
            depth_stencil_attachment={"view": self.depth.create_view(), "depth_clear_value": 1.0,
                                      "depth_load_op": wgpu.LoadOp.clear,
                                      "depth_store_op": wgpu.StoreOp.store})
        buffers = {FRAME: self.frame, POINTS: self.points, VECTORS: self.vectors,
                   TRIANGLES: self.triangles, SCALARS: self.scalars, **self.exact}
        rp.set_pipeline(self.surface)
        rp.set_bind_group(0, self._bind(self.surface, self.surface_bindings,
                                        {b: {"buffer": v} for b, v in buffers.items()}))
        rp.draw(self.count, 1, 0, 0)
        rp.end()
        d.queue.submit([encoder.finish()])
        d.queue.read_buffer(self.ranges[1])         # waits
        timings["surface"] = 1000 * (time.perf_counter() - t0)

        def read(texture, channels, dtype):
            size = np.dtype(dtype).itemsize * channels
            data = d.queue.read_texture({"texture": texture, "origin": (0, 0, 0), "mip_level": 0},
                                        {"offset": 0, "bytes_per_row": w * size}, (w, h, 1))
            return np.frombuffer(data, dtype).reshape(h, w, channels)

        if check:
            return {"values": read(self.field, 4, np.float32), "rows": rows,
                    "timings": timings}

        # LIC, contrast, (LIC, contrast,) composite.
        t0 = time.perf_counter()
        encoder = d.create_command_encoder()
        view = {"field": self.field.create_view(), "color": self.color.create_view(),
                "noise": self.noise.create_view(), "a": self.a.create_view(),
                "b": self.b.create_view(), "final": self.final.create_view()}

        def resources(input_, output=None, range_=0):
            r = {FRAME: {"buffer": self.frame}, FIELD: view["field"], INPUT: view[input_],
                 RANGE: {"buffer": self.ranges[range_]}, COLOR: view["color"],
                 FINAL: view["final"]}
            if output:
                r[OUTPUT] = view[output]
            return r

        self._dispatch(encoder, "lic", resources("noise", "a"))
        self._dispatch(encoder, "histogram", resources("a"))
        self._dispatch(encoder, "percentiles", resources("a"), one=True)
        self._dispatch(encoder, "enhance", resources("a", "b"))
        if enhanced:
            self._dispatch(encoder, "lic", resources("b", "a"))
            self._dispatch(encoder, "histogram", resources("a", range_=1))
            self._dispatch(encoder, "percentiles", resources("a", range_=1), one=True)
            self._dispatch(encoder, "enhance", resources("a", "b", range_=1))
        self._dispatch(encoder, "composite", resources("b"))
        d.queue.submit([encoder.finish()])
        d.queue.read_buffer(self.ranges[1])         # waits
        timings["lic"] = 1000 * (time.perf_counter() - t0)

        final = read(self.final, 4, np.uint8).astype(np.float32)
        s = self.ss
        picture = final.reshape(h // s, s, w // s, s, 4).mean(axis=(1, 3)).round().astype(np.uint8)
        return {"picture": picture, "field": read(self.field, 4, np.float32),
                "noise": read(self.noise, 1, np.float32)[..., 0],
                "lic": read(self.b, 1, np.float32)[..., 0], "timings": timings}


def _box(image, r):
    """A (2r+1)^2 box filter (sums by cumulative sums)."""
    p = np.pad(image, r, mode="edge")
    s = np.zeros((p.shape[0] + 1, p.shape[1] + 1))
    s[1:, 1:] = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return (s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]) / (k * k)


def orientation_error(image, field, radius=6):
    """Per pixel, the angle (degrees, 0 to 90) between the image's direction of least
    change (the structure tensor's minor eigenvector) and the field's screen direction,
    over covered pixels whose window is all covered and where the field is not small."""
    gy, gx = np.gradient(image.astype(np.float64))
    jxx, jyy, jxy = _box(gx * gx, radius), _box(gy * gy, radius), _box(gx * gy, radius)
    major = 0.5 * np.arctan2(2 * jxy, jxx - jyy)       # the gradient's direction
    minor = major + np.pi / 2
    along = np.arctan2(field[..., 1], field[..., 0])
    diff = np.abs((minor - along + np.pi / 2) % np.pi - np.pi / 2)
    covered = (field[..., 2] >= 0).astype(np.float64)
    inside = _box(covered, radius + 1) > 0.999
    strong = field[..., 2] > 0.2 * field[..., 2].max()
    return np.degrees(diff[inside & strong])


def write_png(path, rgba):
    """With VTK: ``rgba`` (``(h, w, 4)`` u8, row 0 at the top)."""
    import vtkmodules.vtkCommonDataModel  # noqa: F401
    from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkIOImage import vtkPNGWriter

    h, w, c = rgba.shape
    array = vtkUnsignedCharArray()
    array.SetNumberOfComponents(c)
    array.SetNumberOfTuples(w * h)
    np.asarray(array)[:] = rgba[::-1].reshape(-1, c)
    image = vtkImageData()
    image.SetDimensions(w, h, 1)
    image.GetPointData().SetScalars(array)
    writer = vtkPNGWriter()
    writer.SetFileName(path)
    writer.SetInputData(image)
    writer.Write()


def gray(values, mask):
    v = np.where(mask, values, 1.0)
    out = np.empty(values.shape + (4,), np.uint8)
    out[..., :3] = (np.clip(v, 0, 1) * 255).round().astype(np.uint8)[..., None]
    out[..., 3] = 255
    return out


def main(out="."):
    tack.set_default_device("cpu")
    t = time.perf_counter()
    points, triangles, vectors, normal, _ = scene.slice_triangles(scene.volume(24))
    print(f"slice: {len(triangles)} triangles, {len(points)} points "
          f"({1000 * (time.perf_counter() - t):.0f} ms with the volume)", flush=True)
    vt = vectors - (vectors @ normal)[:, None] * normal
    speed = np.linalg.norm(vt, axis=1)
    # LIC at the picture's resolution: at 2x it costs 7x as much (four times the
    # pixels, twice the steps) for a slightly smoother slice edge.
    lic = LIC(points, triangles, vectors, os.path.join(out, "wgsl"), size=(1920, 1080),
              supersample=int(os.environ.get("SUPERSAMPLE", "1")))
    centre = np.array([0.5, 0.5, 0.5])
    eye = centre + 1.9 * np.array([0.35, -0.45, 0.82]) / np.linalg.norm([0.35, -0.45, 0.82])
    view = dict(eye=eye, target=centre, up=(0.0, 1.0, 0.0),
                field_range=(float(speed.min()), float(speed.max())))
    lic.render(**view)                                  # warm up
    runs = [lic.render(**view) for _ in range(5)]
    result = min(runs, key=lambda r: r["timings"]["lic"])
    print(f"{lic.w} x {lic.h} ({lic.ss}x of {lic.size[0]} x {lic.size[1]}): surface pass "
          f"{result['timings']['surface']:.1f} ms, LIC passes {result['timings']['lic']:.1f} ms",
          flush=True)
    write_png(os.path.join(out, "lic.png"), result["picture"])
    mask = result["field"][..., 2] >= 0
    write_png(os.path.join(out, "lic_gray.png"), gray(result["lic"], mask))
    write_png(os.path.join(out, "noise.png"), gray(result["noise"], mask))
    for label, image in (("LIC", result["lic"]), ("noise (control)", result["noise"])):
        err = orientation_error(image, result["field"])
        print(f"  {label}: orientation vs the field over {err.size} pixels: median "
              f"{np.median(err):.1f} deg, 90th percentile {np.percentile(err, 90):.1f} deg",
              flush=True)
    print(f"  pictures in {os.path.abspath(out)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
