# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Offscreen WebGPU (wgpu-py) drawing of a surface's launch groups, each a pipeline of
its own WGSL (``shaders.group_shaders``): one instanced draw of a subdivided reference
face per group. Returns the picture and, per pixel, the field, the surface cell and
its (u, v), for checking."""

import os
import sys

import numpy as np
import wgpu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "webgpu_surfaces"))

from renderer import colormap, look_at, perspective, template

QUAD, TRIANGLE = 1, 0


class Renderer:
    def __init__(self):
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.adapter = adapter
        self.device = adapter.request_device_sync()     # WebGPU's default limits
        self._colors = self._storage(colormap())

    def _storage(self, array):
        data = np.ascontiguousarray(array)
        if data.nbytes == 0:
            data = np.zeros(4, np.uint32)
        return self.device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.STORAGE)

    def render(self, groups, bounds, eye, target, field_range, up=(0.0, 0.0, 1.0),
               size=(896, 704), supersample=2, edges=1.2, level=8, tessellation=0.0, fovy=30.0):
        """The picture (``(h, w, 4)`` u8) and probe (``(h, w, 4)`` f32: field, surface
        cell, u, v; cell -1 where none) of ``groups`` from ``eye`` towards ``target``.
        ``level`` subdivides each face's template, ``bounds`` (``(lo, hi)`` corners)
        sets the depth range."""
        w, h = size[0] * supersample, size[1] * supersample
        eye, target = np.asarray(eye, float), np.asarray(target, float)
        radius = float(np.linalg.norm(np.asarray(bounds[1]) - bounds[0])) + 1e-9
        distance = np.linalg.norm(eye - target)
        proj = perspective(fovy, w / h, max(distance - radius, 1e-3 * distance), distance + radius)
        rows = proj @ look_at(eye, target, np.asarray(up, float))
        frame = np.zeros((7, 4), np.float32)
        frame[:4] = rows
        frame[4, :3] = eye
        frame[5] = (field_range[0], field_range[1], 0.0, edges * supersample)
        frame[6, 0] = tessellation * supersample
        frame_buffer = self._storage(frame)

        usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
        color = self.device.create_texture(size=(w, h, 1), format=wgpu.TextureFormat.rgba8unorm,
                                           usage=usage)
        probe = self.device.create_texture(size=(w, h, 1), format=wgpu.TextureFormat.rgba32float,
                                           usage=usage)
        depth = self.device.create_texture(size=(w, h, 1), format=wgpu.TextureFormat.depth32float,
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
        for g in groups:
            module = self.device.create_shader_module(code=g.wgsl)
            pipeline = self.device.create_render_pipeline(
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
            kind = TRIANGLE if g.shape.NUM_POINTS == 3 else QUAD
            verts = template(kind, level)
            pools = [self._storage(data) for _, data in g.buffers]
            mine = [self._storage(verts), self._colors, self._storage(g.ids), frame_buffer]
            bind0 = self.device.create_bind_group(
                layout=pipeline.get_bind_group_layout(0),
                entries=[{"binding": b, "resource": {"buffer": buf}}
                         for (b, _), buf in zip(g.buffers, pools, strict=True)])
            bind1 = self.device.create_bind_group(
                layout=pipeline.get_bind_group_layout(1),
                entries=[{"binding": 0, "resource": {"buffer": mine[3]}},
                         {"binding": 1, "resource": {"buffer": mine[0]}},
                         {"binding": 2, "resource": {"buffer": mine[1]}},
                         {"binding": 3, "resource": {"buffer": mine[2]}}])
            rp.set_pipeline(pipeline)
            rp.set_bind_group(0, bind0)
            rp.set_bind_group(1, bind1)
            rp.draw(len(verts), len(g.ids), 0, 0)
        rp.end()
        self.device.queue.submit([encoder.finish()])

        def read(texture, bytes_per_pixel, dtype):
            data = self.device.queue.read_texture(
                {"texture": texture, "origin": (0, 0, 0), "mip_level": 0},
                {"offset": 0, "bytes_per_row": w * bytes_per_pixel}, (w, h, 1))
            return np.frombuffer(data, dtype).reshape(h, w, 4)

        picture = read(color, 4, np.uint8).astype(np.float32)
        picture = picture.reshape(h // supersample, supersample, w // supersample, supersample, 4)
        return picture.mean(axis=(1, 3)).round().astype(np.uint8), read(probe, 16, np.float32)
