// SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
// SPDX-License-Identifier: BSD-3-Clause

// Image-space surface LIC, every pass in Slang (compiled to WGSL by slangc):
//
// 1. surface (vs_surface, fs_surface): the surface into three targets: the field's
//    tangential part as a unit screen-space direction, its magnitude and the view depth;
//    the colour; a white noise fixed to the surface (a 3D lattice in model space whose
//    cell follows the pixel footprint, two octaves faded so zooming does not pop);
// 2. lic: per covered pixel, the noise averaged along the field's screen-space
//    streamline, forward and backward (RK2, a Hann kernel), stopping at the surface's
//    edge, at depth jumps and where the field vanishes;
// 3. minmax and enhance: a contrast stretch (and a high-pass), between and after two
//    LIC passes (enhanced LIC);
// 4. composite: the colour times the LIC intensity.

struct Frame
{
    float4 row0;     // projection * view, by rows
    float4 row1;
    float4 row2;
    float4 row3;
    float4 eye;      // xyz
    float4 field;    // colour range lo, hi; noise cell in pixels; colour by: 0 |v_t|, 1 the scalar
    float4 size;     // width, height in pixels; step in pixels; steps each way
    float4 lic;      // depth jump (relative); smallest |v_t|; high-pass weight; 0
};

static const float PI = 3.14159265358979f;

float4 clipOf(Frame f, float4 p)
{
    return float4(dot(f.row0, p), dot(f.row1, p), dot(f.row2, p), dot(f.row3, p));
}

// ---------------------------------------------------------------------------------------
// 1. The surface pass.

[vk::binding(0, 0)] StructuredBuffer<Frame> frame;
[vk::binding(10, 0)] StructuredBuffer<float> points;
[vk::binding(11, 0)] StructuredBuffer<float> vectors;
[vk::binding(12, 0)] StructuredBuffer<uint> triangles;
[vk::binding(13, 0)] StructuredBuffer<float> scalars;

struct SurfaceVertex
{
    float4 position : SV_Position;
    float3 model : MODEL;
    float3 vector : VECTOR;
    float scalar : SCALAR;
    nointerpolation float3 normal : NORMAL;
};

float3 pointAt(uint i) { return float3(points[3 * i], points[3 * i + 1], points[3 * i + 2]); }

[shader("vertex")]
SurfaceVertex vs_surface(uint vertex : SV_VertexID)
{
    uint t = vertex / 3, k = vertex % 3;
    uint a = triangles[3 * t], b = triangles[3 * t + 1], c = triangles[3 * t + 2];
    uint i = triangles[3 * t + k];
    float3 p = pointAt(i);
    SurfaceVertex out;
    out.position = clipOf(frame[0], float4(p, 1.f));
    out.model = p;
    out.vector = float3(vectors[3 * i], vectors[3 * i + 1], vectors[3 * i + 2]);
    out.scalar = scalars[i];
    out.normal = normalize(cross(pointAt(b) - pointAt(a), pointAt(c) - pointAt(a)));
    return out;
}

float3 viridis(float t)
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

// A white noise value in [0, 1] per integer lattice cell (pcg3d).
float latticeNoise(int3 cell, uint octave)
{
    uint3 v = uint3(cell) * 1664525u + 1013904223u + octave * 2654435769u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return float(v.x & 0xffffffu) / 16777215.f;
}

struct SurfaceTargets
{
    float4 field : SV_Target0;   // screen direction x, y; |v_t|; view depth. Cleared (0, 0, -1, 0)
    float4 color : SV_Target1;
    float noise : SV_Target2;
};

[shader("fragment")]
SurfaceTargets fs_surface(SurfaceVertex in)
{
    Frame f = frame[0];
    float3 n = in.normal;
    float3 v = in.vector;
    float3 vt = v - dot(v, n) * n;            // the field's tangential part
    // Its screen-space direction: the derivative of the pixel position along vt.
    float4 c = clipOf(f, float4(in.model, 1.f));
    float4 dc = clipOf(f, float4(vt, 0.f));
    float2 dndc = (dc.xy * c.w - c.xy * dc.w) / (c.w * c.w);
    float2 dpix = float2(dndc.x * 0.5f * f.size.x, -dndc.y * 0.5f * f.size.y);
    float len = length(dpix);

    SurfaceTargets out;
    out.field = float4(len > 0.f ? dpix / len : float2(0.f), length(vt), c.w);

    float3 view = normalize(f.eye.xyz - in.model);
    float light = 0.3f + 0.7f * abs(dot(n, view));
    float coloured = f.field.w > 0.5f ? in.scalar : length(vt);
    float t = saturate((coloured - f.field.x) / (f.field.y - f.field.x));
    out.color = float4(viridis(t) * light, 1.f);

    // Noise fixed to the surface: a lattice in model space whose cell is about
    // f.field.z pixels, quantized to octaves and faded between two of them.
    float footprint = max(length(ddx(in.model)), length(ddy(in.model)));
    float level = log2(max(f.field.z * footprint, 1e-9f));
    float l0 = floor(level);
    float fade = level - l0;
    float s0 = exp2(l0), s1 = exp2(l0 + 1.f);
    uint o0 = uint(int(l0) + 64);
    float n0 = latticeNoise(int3(floor(in.model / s0)), o0);
    float n1 = latticeNoise(int3(floor(in.model / s1)), o0 + 1u);
    out.noise = lerp(n0, n1, fade);
    return out;
}

// ---------------------------------------------------------------------------------------
// 2. LIC.

[vk::binding(1, 0)] Texture2D<float4> fieldImage;
[vk::binding(2, 0)] Texture2D<float> inputImage;
[vk::binding(3, 0)] [format("r32f")] WTexture2D<float> outputImage;

// The field bilinearly at x (pixels, centres at +0.5), from covered texels only;
// false where fewer than half the weight is covered.
bool fieldAt(float2 x, out float4 value)
{
    float2 g = x - 0.5f;
    int2 i0 = int2(floor(g));
    float2 w = g - float2(i0);
    int2 size = int2(frame[0].size.xy);
    value = float4(0.f);
    float total = 0.f;
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            int2 i = i0 + int2(dx, dy);
            if (any(i < 0) || any(i >= size))
                continue;
            float4 s = fieldImage.Load(int3(i, 0));
            if (s.z < 0.f)
                continue;
            float wt = (dx == 0 ? 1.f - w.x : w.x) * (dy == 0 ? 1.f - w.y : w.y);
            value += wt * s;
            total += wt;
        }
    if (total < 0.5f)
        return false;
    value /= total;
    float len = length(value.xy);
    value.xy = len > 0.f ? value.xy / len : float2(0.f);
    return len > 0.f;
}

float inputAt(float2 x)
{
    int2 i = clamp(int2(floor(x)), int2(0), int2(frame[0].size.xy) - 1);
    return inputImage.Load(int3(i, 0));
}

// The weighted sum of the input along the streamline from x0, one way.
void stream(float2 x0, float sign, float depth0, inout float sum, inout float weight)
{
    Frame f = frame[0];
    float h = f.size.z;
    int steps = int(f.size.w);
    float2 x = x0;
    float depth = depth0;
    float4 a;
    if (!fieldAt(x, a))
        return;
    for (int i = 1; i <= steps; ++i) {
        // RK2 (midpoint); the field at a step's end starts the next.
        float4 m, b;
        if (a.z < f.lic.y || !fieldAt(x + 0.5f * h * sign * a.xy, m))
            break;
        x += h * sign * m.xy;
        if (!fieldAt(x, b) || abs(b.w - depth) > f.lic.x * depth)
            break;
        depth = b.w;
        a = b;
        float wt = 0.5f + 0.5f * cos(PI * float(i) / float(steps + 1));
        sum += wt * inputAt(x);
        weight += wt;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lic(uint3 id : SV_DispatchThreadID)
{
    int2 size = int2(frame[0].size.xy);
    if (any(int2(id.xy) >= size))
        return;
    float4 here = fieldImage.Load(int3(id.xy, 0));
    if (here.z < 0.f) {
        outputImage.Store(id.xy, 0.f);
        return;
    }
    float2 x0 = float2(id.xy) + 0.5f;
    float sum = inputAt(x0), weight = 1.f;
    stream(x0, 1.f, here.w, sum, weight);
    stream(x0, -1.f, here.w, sum, weight);
    outputImage.Store(id.xy, sum / weight);
}

// ---------------------------------------------------------------------------------------
// 3. Contrast: a histogram of the covered pixels (256 bins over [0, 1], atomics), its
// percentiles (the lowest and highest frame.lic.w of the pixels clipped), then a stretch
// between them and a high-pass.

static const uint BINS = 256;
[vk::binding(4, 0)] RWStructuredBuffer<Atomic<uint>> range;   // BINS counts, then lo, hi bits

[shader("compute")]
[numthreads(8, 8, 1)]
void histogram(uint3 id : SV_DispatchThreadID)
{
    int2 size = int2(frame[0].size.xy);
    if (any(int2(id.xy) >= size) || fieldImage.Load(int3(id.xy, 0)).z < 0.f)
        return;
    float v = saturate(inputImage.Load(int3(id.xy, 0)));
    range[min(uint(v * float(BINS)), BINS - 1)].add(1u);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void percentiles(uint3 id : SV_DispatchThreadID)
{
    uint total = 0;
    for (uint i = 0; i < BINS; ++i)
        total += range[i].load();
    uint clip = uint(frame[0].lic.w * float(total));
    uint below = 0, lo = 0;
    while (lo < BINS - 1 && below + range[lo].load() <= clip)
        below += range[lo++].load();
    uint above = 0, hi = BINS - 1;
    while (hi > lo && above + range[hi].load() <= clip)
        above += range[hi--].load();
    range[BINS].store(asuint(float(lo) / float(BINS)));
    range[BINS + 1].store(asuint(float(hi + 1) / float(BINS)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void enhance(uint3 id : SV_DispatchThreadID)
{
    int2 size = int2(frame[0].size.xy);
    if (any(int2(id.xy) >= size))
        return;
    if (fieldImage.Load(int3(id.xy, 0)).z < 0.f) {
        outputImage.Store(id.xy, 0.f);
        return;
    }
    float lo = asfloat(range[BINS].load()), hi = asfloat(range[BINS + 1].load());
    float scale = hi > lo ? 1.f / (hi - lo) : 0.f;
    float centre = (inputImage.Load(int3(id.xy, 0)) - lo) * scale;
    // High-pass: away from the covered 3 x 3 neighbourhood's mean.
    float sum = 0.f, count = 0.f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int2 i = int2(id.xy) + int2(dx, dy);
            if (any(i < 0) || any(i >= size) || fieldImage.Load(int3(i, 0)).z < 0.f)
                continue;
            sum += (inputImage.Load(int3(i, 0)) - lo) * scale;
            count += 1.f;
        }
    float k = frame[0].lic.z;
    outputImage.Store(id.xy, saturate(centre + k * (centre - sum / max(count, 1.f))));
}

// ---------------------------------------------------------------------------------------
// 4. Composite: the colour times the LIC intensity (VTK's multiply mode); white where no
// surface.

[vk::binding(5, 0)] Texture2D<float4> colorImage;
[vk::binding(6, 0)] [format("rgba8")] WTexture2D<float4> finalImage;

[shader("compute")]
[numthreads(8, 8, 1)]
void composite(uint3 id : SV_DispatchThreadID)
{
    int2 size = int2(frame[0].size.xy);
    if (any(int2(id.xy) >= size))
        return;
    if (fieldImage.Load(int3(id.xy, 0)).z < 0.f) {
        finalImage.Store(id.xy, float4(1.f));
        return;
    }
    float l = inputImage.Load(int3(id.xy, 0));
    float3 c = colorImage.Load(int3(id.xy, 0)).rgb;
    finalImage.Store(id.xy, float4(saturate(c * (0.2f + 1.0f * l)), 1.f));
}
