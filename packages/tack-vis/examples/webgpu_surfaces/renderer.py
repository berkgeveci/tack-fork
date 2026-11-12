# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Offscreen WebGPU (wgpu-py) rendering of ``faces.Records``: one instanced draw per
(shape, level) of a subdivided reference face, the geometry evaluated per vertex and
the field per fragment (``surface.wgsl``). Returns the picture and, per pixel, the
field, the face and its (u, v), for checking."""

import os
import struct
import zlib

import numpy as np
import wgpu
from faces import QUAD

_SHADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "surface.wgsl")


def template(kind, level):
    """A reference face subdivided ``level`` times a side, as a triangle list of (u, v):
    a triangle's ``level^2`` triangles, a quad's ``2 level^2``."""
    n = level
    out = []
    for j in range(n):
        for i in range(n if kind == QUAD else n - j):
            a, b, c, d = (i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)
            out += [a, b, c]
            if kind == QUAD or i + j < n - 1:
                out += [b, d, c]
    return np.array(out, np.float32) / n


def level_of(geometry_order, curved_level=8):
    """The subdivision of a face of geometry ``geometry_order``: none for a straight
    one (the field needs none; it is evaluated per pixel)."""
    return 1 if geometry_order <= 1 else curved_level


def colormap():
    """ParaView's Cool to Warm as 256 RGBA rows, from VTK's diverging colour function."""
    from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

    ctf = vtkColorTransferFunction()
    ctf.SetColorSpaceToDiverging()
    ctf.AddRGBPoint(0.0, 0.231373, 0.298039, 0.752941)
    ctf.AddRGBPoint(1.0, 0.705882, 0.0156863, 0.14902)
    rows = np.ones((256, 4), np.float32)
    for i in range(256):
        rows[i, :3] = ctf.GetColor(i / 255.0)
    return rows


def look_at(eye, target, up):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fovy, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy) / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, far / (near - far), near * far / (near - far)], [0, 0, -1, 0]])


def write_png(path, rgba):
    """``rgba`` (``(h, w, 4)`` u8) as a PNG, by zlib alone."""
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


class Renderer:
    def __init__(self):
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        limits = ("max-storage-buffer-binding-size", "max-buffer-size")
        self.device = adapter.request_device_sync(
            required_limits={k: adapter.limits[k] for k in limits})
        self.adapter = adapter
        with open(_SHADER) as f:
            module = self.device.create_shader_module(code=f.read())
        self.pipeline = self.device.create_render_pipeline(
            layout="auto",
            vertex={"module": module, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list,
                       "cull_mode": wgpu.CullMode.none},
            depth_stencil={"format": wgpu.TextureFormat.depth32float,
                           "depth_write_enabled": True,
                           "depth_compare": wgpu.CompareFunction.less},
            fragment={"module": module, "entry_point": "fs_main",
                      "targets": [{"format": wgpu.TextureFormat.rgba8unorm},
                                  {"format": wgpu.TextureFormat.rgba32float}]})

    def _storage(self, array):
        data = np.ascontiguousarray(array)
        if data.nbytes == 0:
            data = np.zeros(4, np.float32)
        return self.device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.STORAGE)

    def render(self, records, eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0), size=(896, 704),
               supersample=2, mode=0, edges=1.2, curved_level=8, fovy=30.0):
        """The picture (``(h, w, 4)`` u8) and the probe (``(h, w, 4)`` f32: field, face,
        u, v; face -1 where no face is) of ``records`` from ``eye`` (world
        coordinates) towards ``target``. ``mode`` 1 colours each face by its field's
        order."""
        w, h = size[0] * supersample, size[1] * supersample
        origin = records.origin
        eye = np.asarray(eye, float) - origin
        target = np.asarray(target, float) - origin
        radius = float(np.abs(records.geometry).max()) * 2.0 + 1e-9
        distance = np.linalg.norm(eye - target)
        proj = perspective(fovy, w / h, max(distance - radius, 1e-3 * distance),
                           distance + radius)
        view_proj = proj @ look_at(eye, target, np.asarray(up, float))
        lo, hi = records.range
        uniforms = np.concatenate([view_proj.T.reshape(-1), [*eye, 1.0],
                                   [lo, hi, float(mode), edges * supersample]]).astype(np.float32)

        # Faces sorted by (shape, level), one draw each over their run.
        kinds = records.faces[:, 0] & 0xFF
        levels = np.array([level_of((x >> 8) & 0xFF, curved_level) for x in records.faces[:, 0]],
                          np.int64)
        order = np.lexsort((levels, kinds))
        faces = records.faces[order]
        runs, verts, base = [], [], 0
        keys = np.stack([kinds[order], levels[order]], axis=1)
        for kind, level in sorted({tuple(k) for k in keys.tolist()}):
            selected = np.flatnonzero((keys[:, 0] == kind) & (keys[:, 1] == level))
            t = template(kind, level)
            runs.append((base, len(t), int(selected[0]), len(selected)))
            verts.append(t)
            base += len(t)
        templates = np.concatenate(verts) if verts else np.zeros((1, 2), np.float32)

        buffers = [self.device.create_buffer_with_data(
                       data=uniforms, usage=wgpu.BufferUsage.UNIFORM),
                   self._storage(faces), self._storage(records.geometry.reshape(-1)),
                   self._storage(records.field), self._storage(templates),
                   self._storage(colormap())]
        bind = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
                     for i, b in enumerate(buffers)])

        usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
        color = self.device.create_texture(size=(w, h, 1), format=wgpu.TextureFormat.rgba8unorm,
                                           usage=usage)
        probe = self.device.create_texture(size=(w, h, 1),
                                           format=wgpu.TextureFormat.rgba32float, usage=usage)
        depth = self.device.create_texture(size=(w, h, 1),
                                           format=wgpu.TextureFormat.depth32float,
                                           usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
        encoder = self.device.create_command_encoder()
        rp = encoder.begin_render_pass(
            color_attachments=[
                {"view": color.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store, "clear_value": (1, 1, 1, 1)},
                {"view": probe.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store, "clear_value": (0, -1, 0, 0)}],
            depth_stencil_attachment={"view": depth.create_view(), "depth_clear_value": 1.0,
                                      "depth_load_op": wgpu.LoadOp.clear,
                                      "depth_store_op": wgpu.StoreOp.store})
        rp.set_pipeline(self.pipeline)
        rp.set_bind_group(0, bind)
        for first_vertex, count, first_face, faces_in_run in runs:
            rp.draw(count, faces_in_run, first_vertex, first_face)
        rp.end()
        self.device.queue.submit([encoder.finish()])

        def read(texture, bytes_per_pixel, dtype):
            data = self.device.queue.read_texture(
                {"texture": texture, "origin": (0, 0, 0), "mip_level": 0},
                {"offset": 0, "bytes_per_row": w * bytes_per_pixel}, (w, h, 1))
            return np.frombuffer(data, dtype).reshape(h, w, 4)

        picture = read(color, 4, np.uint8).astype(np.float32)
        picture = picture.reshape(h // supersample, supersample, w // supersample, supersample, 4)
        picture = picture.mean(axis=(1, 3)).round().astype(np.uint8)
        values = read(probe, 16, np.float32).copy()
        # The probe's face is the sorted position: back to the records' own.
        hit = values[..., 1] >= 0
        values[..., 1][hit] = order[values[..., 1][hit].astype(np.int64)]
        return picture, values
