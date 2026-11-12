# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""WebGPU (wgpu-py) frame times of the two Tack surface renderers on the same surfaces:

- ``records``: the WebGPU prototype (``webgpu_surfaces``), each face's Bernstein
  coefficients and a hand-written WGSL evaluator;
- ``slang``: Tack's own evaluation compiled through Slang (``slang_surfaces``), one
  pipeline a launch group;
- ``cached``: the same, with the geometry evaluated once by Tack at the template's
  points (position and normal per point) and drawn indexed, not instanced, so a vertex
  reads its point; the field still evaluated per pixel;
- ``flat``: ``cached`` without the field (a constant colour), to time the rest.

Each sets up once (Tack's preparation, buffers, pipelines), then draws ``frames``
frames at 1920 x 1080 with the camera orbiting, waiting for the GPU after each:
prints setup, first frame and the median steady frame, in ms. ``BENCH_RENDERERS``
picks them (comma separated; all by default).

    SLANGC=... python bench_wgpu.py OUTDIR [straight|curved] [n ...]
"""

import os
import re
import sys
import time

import numpy as np
import wgpu

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("slang_surfaces", "webgpu_surfaces"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))

from faces import QUAD, _bernstein, surface_records
from renderer import colormap, level_of, look_at, perspective, template
from scenes import camera, indexed_template, surface
from shaders import _geometry_at, group_shaders

import tack

W, H = 1920, 1080


class _Target:
    def __init__(self, device):
        usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
        self.color = device.create_texture(size=(W, H, 1), format=wgpu.TextureFormat.rgba8unorm,
                                           usage=usage)
        self.probe = device.create_texture(size=(W, H, 1), format=wgpu.TextureFormat.rgba32float,
                                           usage=usage)
        self.depth = device.create_texture(size=(W, H, 1), format=wgpu.TextureFormat.depth32float,
                                           usage=wgpu.TextureUsage.RENDER_ATTACHMENT)

    def begin(self, encoder):
        return encoder.begin_render_pass(
            color_attachments=[
                {"view": self.color.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store, "clear_value": (1, 1, 1, 1)},
                {"view": self.probe.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store, "clear_value": (0, -1, 0, 0)}],
            depth_stencil_attachment={"view": self.depth.create_view(), "depth_clear_value": 1.0,
                                      "depth_load_op": wgpu.LoadOp.clear,
                                      "depth_store_op": wgpu.StoreOp.store})


def _view_proj(eye, target):
    proj = perspective(30.0, W / H, 0.5, 6.0)
    return proj @ look_at(np.asarray(eye, float), np.asarray(target, float),
                          np.array([0.0, 0.0, 1.0]))


def _pipeline(device, wgsl, vs, fs):
    module = device.create_shader_module(code=wgsl)
    return device.create_render_pipeline(
        layout="auto", vertex={"module": module, "entry_point": vs, "buffers": []},
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list,
                   "cull_mode": wgpu.CullMode.none},
        depth_stencil={"format": wgpu.TextureFormat.depth32float, "depth_write_enabled": True,
                       "depth_compare": wgpu.CompareFunction.less},
        fragment={"module": module, "entry_point": fs,
                  "targets": [{"format": wgpu.TextureFormat.rgba8unorm},
                              {"format": wgpu.TextureFormat.rgba32float}]})


def _storage(device, array, usage=wgpu.BufferUsage.STORAGE):
    data = np.ascontiguousarray(array)
    return device.create_buffer_with_data(data=data if data.nbytes else np.zeros(4, np.uint32),
                                          usage=usage)


class Records:
    """The WebGPU prototype: Bernstein records, ``surface.wgsl``."""

    def __init__(self, device, surf, level):
        self.device = device
        self.surface = surf
        self.records = r = surface_records(surf, "f")
        kinds = r.faces[:, 0] & 0xFF
        order = np.argsort(kinds, kind="stable")
        faces = r.faces[order]
        self.runs, verts, base = [], [], 0
        for kind in sorted(set(kinds.tolist())):
            sel = np.flatnonzero(kinds[order] == kind)
            g = (faces[sel[0], 0] >> 8) & 0xFF
            t = template(kind, level_of(g, level))
            self.runs.append((base, len(t), int(sel[0]), len(sel)))
            verts.append(t)
            base += len(t)
        with open(os.path.join(HERE, "..", "webgpu_surfaces", "surface.wgsl")) as f:
            self.pipeline = _pipeline(device, f.read(), "vs_main", "fs_main")
        self.uniforms = device.create_buffer(size=4 * 28, usage=wgpu.BufferUsage.UNIFORM
                                             | wgpu.BufferUsage.COPY_DST)
        self.field = _storage(device, r.field, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        buffers = [self.uniforms, _storage(device, faces), _storage(device, r.geometry.reshape(-1)),
                   self.field, _storage(device, np.concatenate(verts)),
                   _storage(device, colormap())]
        self.bind = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[{"binding": i, "resource": {"buffer": b}} for i, b in enumerate(buffers)])
        self.triangles = sum(c // 3 * n for _, c, _, n in self.runs)

    def update(self, field):
        """A new field of the same space: its Bernstein coefficients, uploaded."""
        values, _ = _bernstein(self.surface, field, field.space.order)
        self.device.queue.write_buffer(self.field, 0, values.reshape(-1).astype(np.float32))

    def draw(self, target, eye, centre):
        eye = np.asarray(eye) - self.records.origin
        vp = _view_proj(eye, np.asarray(centre) - self.records.origin)
        lo, hi = self.records.range
        u = np.concatenate([vp.T.reshape(-1), [*eye, 1.0], [lo, hi, 0.0, 1.2], [0, 0, 0, 0]])
        self.device.queue.write_buffer(self.uniforms, 0, u.astype(np.float32))
        encoder = self.device.create_command_encoder()
        rp = target.begin(encoder)
        rp.set_pipeline(self.pipeline)
        rp.set_bind_group(0, self.bind)
        for first_vertex, count, first_face, n in self.runs:
            rp.draw(count, n, first_vertex, first_face)
        rp.end()
        self.device.queue.submit([encoder.finish()])


class Slang:
    """Tack's own evaluation through Slang, a pipeline a launch group."""

    def __init__(self, device, surf, level, out_dir):
        self.device = device
        groups = group_shaders(surf, "f", out_dir)
        values = surf.fields["f"].values.numpy().reshape(-1)
        self.range = (float(values.min()), float(values.max()))
        curved = surf.geometry.space.order > 1
        self.frame = device.create_buffer(size=16 * 7, usage=wgpu.BufferUsage.STORAGE
                                          | wgpu.BufferUsage.COPY_DST)
        colors = _storage(device, colormap())
        self.draws, self.triangles, self.fields = [], 0, []
        for g in groups:
            pipeline = _pipeline(device, g.wgsl, "vs_main", "fs_main")
            verts = template(QUAD if g.shape.NUM_POINTS == 4 else 0, level if curved else 1)
            pools = [_storage(device, data, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST) for _, data in g.buffers]
            self.fields.append((pools[-1], _place(g.buffers[-1][1], surf.fields["f"])))
            bind0 = device.create_bind_group(
                layout=pipeline.get_bind_group_layout(0),
                entries=[{"binding": b, "resource": {"buffer": p}}
                         for (b, _), p in zip(g.buffers, pools, strict=True)])
            mine = [self.frame, _storage(device, verts), colors, _storage(device, g.ids)]
            bind1 = device.create_bind_group(
                layout=pipeline.get_bind_group_layout(1),
                entries=[{"binding": i, "resource": {"buffer": b}} for i, b in enumerate(mine)])
            self.draws.append((pipeline, bind0, bind1, len(verts), len(g.ids)))
            self.triangles += len(verts) // 3 * len(g.ids)

    def update(self, field):
        """A new field of the same space: its values, in their own basis, uploaded where
        the generated code reads them."""
        values = field.values.numpy().reshape(-1).astype(np.float32)
        for pool, at in self.fields:
            self.device.queue.write_buffer(pool, 4 * at, values)

    def draw(self, target, eye, centre):
        frame = np.zeros((7, 4), np.float32)
        frame[:4] = _view_proj(eye, centre)
        frame[4, :3] = eye
        frame[5] = (*self.range, 0.0, 1.2)
        self.device.queue.write_buffer(self.frame, 0, frame)
        encoder = self.device.create_command_encoder()
        rp = target.begin(encoder)
        for pipeline, bind0, bind1, count, n in self.draws:
            rp.set_pipeline(pipeline)
            rp.set_bind_group(0, bind0)
            rp.set_bind_group(1, bind1)
            rp.draw(count, n, 0, 0)
        rp.end()
        self.device.queue.submit([encoder.finish()])


def _place(pool, field):
    """Where ``field``'s values start in a generated function's pool (in words): the
    pool holds the view's arrays narrowed to 32 bits, one after another."""
    words = field.values.numpy().reshape(-1).astype(np.float32).view(np.uint32)
    for at in np.flatnonzero(pool[:len(pool) - len(words) + 1] == words[0]):
        if np.array_equal(pool[at:at + len(words)], words):
            return int(at)
    raise ValueError("the field's values are not in the pool")


def _used(wgsl, buffers, group=0):
    """The (binding, data) of ``buffers`` the shader declares (Slang drops unused ones)."""
    declared = {int(b) for b in re.findall(rf"@binding\((\d+)\) @group\({group}\)", wgsl)}
    return [(b, d) for b, d in buffers if b in declared]


class Cached:
    """Tack + Slang with the geometry evaluated once (``flat``: without the field)."""

    def __init__(self, device, surf, level, out_dir, flat=False):
        self.device = device
        defines = ("TACK_CACHED", "TACK_FLAT") if flat else ("TACK_CACHED",)
        groups = group_shaders(surf, "f", out_dir, defines)
        values = surf.fields["f"].values.numpy().reshape(-1)
        self.range = (float(values.min()), float(values.max()))
        curved = surf.geometry.space.order > 1
        self.frame = device.create_buffer(size=16 * 7, usage=wgpu.BufferUsage.STORAGE
                                          | wgpu.BufferUsage.COPY_DST)
        colors = _storage(device, colormap())
        self.draws, self.triangles, self.fields = [], 0, []
        for g in groups:
            pipeline = _pipeline(device, g.wgsl, "vs_main", "fs_main")
            points, tris = indexed_template(g.shape.NUM_POINTS == 4,
                                            level if curved else 1)
            count, m = len(g.ids), len(points)
            # The geometry at every cell's points, by Tack (here on the CPU).
            ids = tack.array(np.repeat(np.arange(count, dtype=np.int32), m))
            pcs = np.zeros((count * m, 3), np.float32)
            pcs[:, :2] = np.tile(points, (count, 1))
            geo = tack.empty((3 * count * m,), tack.vec3())
            _geometry_at(surf.domain_view("cells", g.group), ids, tack.array(pcs, tack.vec3()),
                         geo)
            geo = geo.numpy().reshape(-1, 3, 3)
            cached = np.zeros((count * m, 2, 4), np.float32)
            cached[:, 0, :3] = geo[:, 0]
            cached[:, 1, :3] = np.cross(geo[:, 1], geo[:, 2])
            index = (tris[None, :] + (m * np.arange(count, dtype=np.uint32))[:, None]).reshape(-1)
            used = [(b, _storage(device, d, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)) for b, d in _used(g.wgsl, g.buffers)]
            if not flat:
                self.fields.append((used[-1][1], _place(g.buffers[-1][1], surf.fields["f"])))
            entries = [{"binding": b, "resource": {"buffer": u}} for b, u in used]
            bind0 = device.create_bind_group(layout=pipeline.get_bind_group_layout(0),
                                             entries=entries) if entries else None
            mine = _used(g.wgsl, enumerate([
                self.frame, _storage(device, points), colors, _storage(device, g.ids),
                _storage(device, cached.reshape(-1, 4)),
                _storage(device, np.array([m, 0, 0, 0], np.uint32))]), group=1)
            bind1 = device.create_bind_group(
                layout=pipeline.get_bind_group_layout(1),
                entries=[{"binding": i, "resource": {"buffer": b}} for i, b in mine])
            indices = _storage(device, index, wgpu.BufferUsage.INDEX)
            self.draws.append((pipeline, bind0, bind1, indices, len(index)))
            self.triangles += len(index) // 3

    def update(self, field):
        """A new field of the same space: its values, in their own basis, uploaded where
        the generated code reads them."""
        values = field.values.numpy().reshape(-1).astype(np.float32)
        for pool, at in self.fields:
            self.device.queue.write_buffer(pool, 4 * at, values)

    def draw(self, target, eye, centre):
        frame = np.zeros((7, 4), np.float32)
        frame[:4] = _view_proj(eye, centre)
        frame[4, :3] = eye
        frame[5] = (*self.range, 0.0, 1.2)
        self.device.queue.write_buffer(self.frame, 0, frame)
        encoder = self.device.create_command_encoder()
        rp = target.begin(encoder)
        for pipeline, bind0, bind1, indices, count in self.draws:
            rp.set_pipeline(pipeline)
            if bind0 is not None:
                rp.set_bind_group(0, bind0)
            rp.set_bind_group(1, bind1)
            rp.set_index_buffer(indices, wgpu.IndexFormat.uint32)
            rp.draw_indexed(count, 1, 0, 0, 0)
        rp.end()
        self.device.queue.submit([encoder.finish()])


class Flat(Cached):
    def __init__(self, device, surf, level, out_dir):
        super().__init__(device, surf, level, out_dir, flat=True)


def bench(device, make, surf, frames=None):
    frames = frames or int(os.environ.get("BENCH_FRAMES", 30))
    target = _Target(device)
    start = time.perf_counter()
    r = make()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    r.draw(target, *camera(0, frames))
    device.queue.on_submitted_work_done_sync()
    first = time.perf_counter() - start
    times = []
    for k in range(1, frames + 1):
        start = time.perf_counter()
        r.draw(target, *camera(k, frames))
        device.queue.on_submitted_work_done_sync()
        times.append(time.perf_counter() - start)
    if os.environ.get("BENCH_PNG"):
        from renderer import write_png

        data = device.queue.read_texture(
            {"texture": target.color, "origin": (0, 0, 0), "mip_level": 0},
            {"offset": 0, "bytes_per_row": W * 4}, (W, H, 1))
        name = f"{type(r).__name__.lower()}-{os.environ.get('BENCH_TAG', 'wgpu')}-{len(r.__dict__)}"
        write_png(os.path.join(os.environ["BENCH_PNG"], f"{name}-{r.triangles}.png"),
                  np.frombuffer(data, np.uint8).reshape(H, W, 4))
    return r.triangles, setup, first, float(np.median(times))


def open_device():
    """The adapter BENCH_ADAPTER names (a substring of its device's name), else the
    high-performance one, and a device with its storage limits."""
    wanted = os.environ.get("BENCH_ADAPTER")
    if wanted:
        adapters = wgpu.gpu.enumerate_adapters_sync()
        adapter = next((a for a in adapters if wanted.lower() in a.info["device"].lower()), None)
        if adapter is None:
            raise RuntimeError(f"no adapter {wanted!r} in "
                               f"{[(a.info['backend_type'], a.info['device']) for a in adapters]}")
    else:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync(required_limits={
        k: adapter.limits[k] for k in ("max-storage-buffer-binding-size", "max-buffer-size")})
    return adapter, device


def makers(device, out, level=8):
    return {"records": lambda s: Records(device, s, level),
            "slang": lambda s: Slang(device, s, level, out),
            "cached": lambda s: Cached(device, s, level, out),
            "flat": lambda s: Flat(device, s, level, out)}


def first(renderer, surf, out):
    """A fresh process's first picture: the device, the renderer's setup and the first
    frame (drawn, waited for and read back), in ms."""
    start = time.perf_counter()
    _, device = open_device()
    times = {"device": time.perf_counter() - start}
    target = _Target(device)
    start = time.perf_counter()
    r = makers(device, out)[renderer](surf)
    times["setup"] = time.perf_counter() - start
    start = time.perf_counter()
    r.draw(target, *camera(0, 30))
    device.queue.read_texture({"texture": target.color, "origin": (0, 0, 0), "mip_level": 0},
                              {"offset": 0, "bytes_per_row": W * 4}, (W, H, 1))
    times["first"] = time.perf_counter() - start
    return {k: 1e3 * v for k, v in times.items()}


def main(out, kind="straight", sizes=(16, 32, 64, 128), level=8):
    tack.set_default_device("cpu")
    adapter, device = open_device()
    info = adapter.info
    print(f"# wgpu {info['backend_type']} {info['device']} ({info['adapter_type']}), {W}x{H}")
    print("renderer kind n faces triangles setup_ms first_ms frame_ms")
    for n in sizes:
        start = time.perf_counter()
        surf = surface(kind, n)
        made = time.perf_counter() - start
        print(f"# {kind} n={n}: {surf.num_cells} faces, Tack's surface in {made:.1f} s")
        make = makers(device, out, level)
        for name in os.environ.get("BENCH_RENDERERS", ",".join(make)).split(","):
            tri, setup, first_frame, frame = bench(device, lambda s=surf, m=make[name]: m(s), surf)
            print(f"{name} {kind} {n} {surf.num_cells} {tri} {1e3 * setup:.0f} "
                  f"{1e3 * first_frame:.1f} {1e3 * frame:.2f}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], args[1] if len(args) > 1 else "straight",
         tuple(int(a) for a in args[2:]) or (16, 32, 64, 128))
