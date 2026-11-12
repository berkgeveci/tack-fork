// SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
// SPDX-License-Identifier: BSD-3-Clause

// The renderer's half of a launch group's shader. Above this file in the same
// source are Tack's generated functions:
//
//   float3x3 tack_geometry(int cell, float3 pc)  // rows: position, d/dr, d/ds
//   float    tack_value(int cell, float3 pc)     // the field
//
// with `cell` the group's own cell index and `pc` its parametric point. One instanced
// draw per group: vertex_index picks a vertex of a subdivided reference face (the
// template), instance_index the cell. The fragment shader evaluates the field at the
// interpolated parametric point, exact on every template triangle.

// Group 1: the renderer's; group 0 is the generated functions'.
[[vk::binding(0, 1)]] StructuredBuffer<float4> frame;      // view-projection rows 0-3, eye,
                                                          // (lo, hi, unused, edge px),
                                                          // (template px, ...)
[[vk::binding(1, 1)]] StructuredBuffer<float2> templates;
[[vk::binding(2, 1)]] StructuredBuffer<float4> colors;     // 256 rows of a colour map
[[vk::binding(3, 1)]] StructuredBuffer<uint> cell_ids;     // the group's cells in the surface

struct Varyings
{
    float4 position : SV_Position;
    float2 uv : UV;
    nointerpolation uint cell : CELL;
    float3 normal : NORMAL;
    float3 world : WORLD;
    float3 corner : CORNER;      // the template triangle's barycentric coordinates
};

[shader("vertex")]
Varyings vs_main(uint vertex : SV_VertexID, uint instance : SV_InstanceID)
{
    float2 uv = templates[vertex];
    float3x3 g = tack_geometry(int(instance), float3(uv, 0.0));
    float4 p = float4(g[0], 1.0);
    Varyings out;
    out.position = float4(dot(frame[0], p), dot(frame[1], p), dot(frame[2], p), dot(frame[3], p));
    out.uv = uv;
    out.cell = instance;
    out.normal = cross(g[1], g[2]);
    out.world = g[0];
    uint which = vertex % 3;
    out.corner = float3(which == 0 ? 1.0 : 0.0, which == 1 ? 1.0 : 0.0, which == 2 ? 1.0 : 0.0);
    return out;
}

struct Targets
{
    float4 color : SV_Target0;
    float4 probe : SV_Target1;   // the field, the surface cell, u, v: for checking
};

[shader("fragment")]
Targets fs_main(Varyings v)
{
    // Derivatives first, in uniform control flow.
    float fu = fwidth(v.uv.x);
    float fv = fwidth(v.uv.y);
    float fw = fwidth(1.0 - v.uv.x - v.uv.y);
    float3 fc = fwidth(v.corner);

    float value = tack_value(int(v.cell), float3(v.uv, 0.0));

    float lo = frame[5].x;
    float hi = frame[5].y;
    float t = clamp((value - lo) / max(hi - lo, 1e-30), 0.0, 1.0);
    float at = t * 255.0;
    uint i = uint(floor(at));
    float3 base = lerp(colors[i].rgb, colors[min(i + 1, 255)].rgb, at - float(i));

    float3 n = normalize(v.normal);
    float3 view = normalize(frame[4].xyz - v.world);
    if (dot(n, view) < 0.0)
        n = -n;
    float diffuse = max(dot(n, view), 0.0);
    float3 color = base * (0.3 + 0.7 * diffuse) + 0.15 * pow(diffuse, 40.0);

    float width = frame[5].w;
    if (width > 0.0)
    {
        float e = min(v.uv.x / fu, v.uv.y / fv);
#if TACK_TRIANGLE
        e = min(e, (1.0 - v.uv.x - v.uv.y) / fw);
#else
        e = min(e, min((1.0 - v.uv.x) / fu, (1.0 - v.uv.y) / fv));
#endif
        float line = 1.0 - smoothstep(width - 0.5, width + 0.5, e);
        color = lerp(color, float3(0.1, 0.1, 0.12), line);
    }
    float thin = frame[6].x;
    if (thin > 0.0)
    {
        float3 d3 = v.corner / max(fc, float3(1e-12));
        float d = min(d3.x, min(d3.y, d3.z));
        float line = 1.0 - smoothstep(thin - 0.5, thin + 0.5, d);
        color = lerp(color, float3(1.0), 0.8 * line);
    }

    Targets out;
    out.color = float4(color, 1.0);
    out.probe = float4(value, float(cell_ids[v.cell]), v.uv.x, v.uv.y);
    return out;
}
