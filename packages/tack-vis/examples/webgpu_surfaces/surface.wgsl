// SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
// SPDX-License-Identifier: BSD-3-Clause

// High-order surface faces, each with its own Bernstein coefficients (faces.py).
// One draw per (shape, level): vertex_index picks a vertex of the level's template
// (a subdivided reference triangle or quad), instance_index the face. The vertex
// shader places the template's (u, v) by the face's geometry; the fragment shader
// evaluates the field at the interpolated (u, v), which is exact on every template
// triangle, so the colour is the field's at every pixel whatever the level.

const MAXP: u32 = 8u;                  // the highest order a face may have
const TRIANGLE: u32 = 0u;
const FACTORIAL = array<f32, 9>(1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0);

struct Uniforms {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,           // the camera, about the records' origin
    range: vec4<f32>,         // the field's least and greatest, the mode, the edge width
    extra: vec4<f32>,         // the template triangles' edge width (0: none)
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> faces: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> geometry: array<f32>;
@group(0) @binding(3) var<storage, read> field: array<f32>;
@group(0) @binding(4) var<storage, read> templates: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> colors: array<vec4<f32>>;

// A face's basis functions at (u, v) and their derivatives, in its coefficients' order.
struct Basis {
    count: u32,
    w: array<f32, 81>,
    du: array<f32, 81>,
    dv: array<f32, 81>,
};

// The degree-n Bernstein polynomials at t, and their derivatives (from degree n - 1).
fn bernstein1(n: u32, t: f32, b: ptr<function, array<f32, 9>>, d: ptr<function, array<f32, 9>>) {
    for (var j = 0u; j <= MAXP; j++) {
        (*b)[j] = 0.0;
        (*d)[j] = 0.0;
    }
    (*b)[0] = 1.0;
    let s = 1.0 - t;
    for (var k = 1u; k <= n; k++) {
        if (k == n) {
            for (var j = 0u; j <= n; j++) {
                var lo = 0.0;
                if (j > 0u) { lo = (*b)[j - 1u]; }
                var hi = 0.0;
                if (j < n) { hi = (*b)[j]; }
                (*d)[j] = f32(n) * (lo - hi);
            }
        }
        for (var j = k; j >= 1u; j--) {
            (*b)[j] = s * (*b)[j] + t * (*b)[j - 1u];
        }
        (*b)[0] = s * (*b)[0];
    }
}

// A quad's coefficient (a, b): its tensor lattice, except at order 1, its corners.
fn quad_index(n: u32, a: u32, b: u32) -> u32 {
    if (n == 1u) {
        let k = a + 2u * b;
        return select(k, 5u - k, k >= 2u);
    }
    return a + (n + 1u) * b;
}

fn basis(kind: u32, n: u32, uv: vec2<f32>) -> Basis {
    var out: Basis;
    if (kind == TRIANGLE) {
        let w = 1.0 - uv.x - uv.y;
        var pr: array<f32, 9>;
        var ps: array<f32, 9>;
        var pw: array<f32, 9>;
        pr[0] = 1.0;
        ps[0] = 1.0;
        pw[0] = 1.0;
        for (var i = 1u; i <= n; i++) {
            pr[i] = pr[i - 1u] * uv.x;
            ps[i] = ps[i - 1u] * uv.y;
            pw[i] = pw[i - 1u] * w;
        }
        var fact = FACTORIAL;            // a var: indexed at run time
        var k = 0u;
        for (var j = 0u; j <= n; j++) {
            for (var i = 0u; i <= n - j; i++) {
                let m = n - i - j;
                let c = fact[n] / (fact[i] * fact[j] * fact[m]);
                out.w[k] = c * pr[i] * ps[j] * pw[m];
                var dr = 0.0;
                var ds = 0.0;
                if (i > 0u) { dr += f32(i) * pr[i - 1u] * ps[j] * pw[m]; }
                if (j > 0u) { ds += f32(j) * pr[i] * ps[j - 1u] * pw[m]; }
                if (m > 0u) {
                    let dm = f32(m) * pr[i] * ps[j] * pw[m - 1u];
                    dr -= dm;
                    ds -= dm;
                }
                out.du[k] = c * dr;
                out.dv[k] = c * ds;
                k++;
            }
        }
        out.count = k;
        return out;
    }
    var bu: array<f32, 9>;
    var du: array<f32, 9>;
    var bv: array<f32, 9>;
    var dv: array<f32, 9>;
    bernstein1(n, uv.x, &bu, &du);
    bernstein1(n, uv.y, &bv, &dv);
    for (var b = 0u; b <= n; b++) {
        for (var a = 0u; a <= n; a++) {
            let k = quad_index(n, a, b);
            out.w[k] = bu[a] * bv[b];
            out.du[k] = du[a] * bv[b];
            out.dv[k] = bu[a] * dv[b];
        }
    }
    out.count = (n + 1u) * (n + 1u);
    return out;
}

struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) face: u32,
    @location(2) normal: vec3<f32>,
    @location(3) world: vec3<f32>,
    @location(4) corner: vec3<f32>,   // the template triangle's barycentric coordinates
};

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) face: u32) -> Varyings {
    let record = faces[face];
    let kind = record.x & 0xFFu;
    let order = (record.x >> 8u) & 0xFFu;
    let uv = templates[vertex];
    let b = basis(kind, order, uv);
    var p = vec3<f32>(0.0);
    var pu = vec3<f32>(0.0);
    var pv = vec3<f32>(0.0);
    for (var k = 0u; k < b.count; k++) {
        let at = 3u * (record.y + k);
        let x = vec3<f32>(geometry[at], geometry[at + 1u], geometry[at + 2u]);
        p += b.w[k] * x;
        pu += b.du[k] * x;
        pv += b.dv[k] * x;
    }
    var out: Varyings;
    out.position = uniforms.view_proj * vec4<f32>(p, 1.0);
    out.uv = uv;
    out.face = face;
    out.normal = cross(pu, pv);
    out.world = p;
    let which = vertex % 3u;
    out.corner = vec3<f32>(f32(which == 0u), f32(which == 1u), f32(which == 2u));
    return out;
}

struct Targets {
    @location(0) color: vec4<f32>,
    @location(1) probe: vec4<f32>,     // the field, the face, u, v: for checking
};

// The categorical colour of an order (mode 1).
fn order_color(order: u32) -> vec3<f32> {
    var table = array<vec3<f32>, 9>(
        vec3<f32>(0.6, 0.6, 0.6), vec3<f32>(0.27, 0.47, 0.67), vec3<f32>(0.4, 0.8, 0.93),
        vec3<f32>(0.13, 0.53, 0.2), vec3<f32>(0.8, 0.73, 0.27), vec3<f32>(0.93, 0.4, 0.47),
        vec3<f32>(0.67, 0.2, 0.47), vec3<f32>(0.73, 0.73, 0.73), vec3<f32>(0.2, 0.2, 0.2));
    return table[min(order, 8u)];
}

@fragment
fn fs_main(in: Varyings) -> Targets {
    // Derivatives first, in uniform control flow (before anything per face).
    let fu = fwidth(in.uv.x);
    let fv = fwidth(in.uv.y);
    let fw = fwidth(1.0 - in.uv.x - in.uv.y);
    let fc = fwidth(in.corner);
    let record = faces[in.face];
    let kind = record.x & 0xFFu;
    let order = record.x >> 16u;
    let b = basis(kind, order, in.uv);
    var value = 0.0;
    for (var k = 0u; k < b.count; k++) {
        value += b.w[k] * field[record.z + k];
    }

    let lo = uniforms.range.x;
    let hi = uniforms.range.y;
    var base: vec3<f32>;
    if (uniforms.range.z > 0.5) {
        base = order_color(order);
    } else {
        let t = clamp((value - lo) / max(hi - lo, 1e-30), 0.0, 1.0);
        let at = t * 255.0;
        let i = u32(floor(at));
        base = mix(colors[i].rgb, colors[min(i + 1u, 255u)].rgb, at - f32(i));
    }

    // A headlight, both sides lit.
    var n = normalize(in.normal);
    let view = normalize(uniforms.eye.xyz - in.world);
    if (dot(n, view) < 0.0) { n = -n; }
    let diffuse = max(dot(n, view), 0.0);
    let specular = pow(max(dot(n, view), 0.0), 40.0);
    var color = base * (0.3 + 0.7 * diffuse) + vec3<f32>(0.15 * specular);

    // The face's edges: where (u, v) is within a few pixels of the reference
    // boundary, by its screen-space rate of change.
    let width = uniforms.range.w;
    if (width > 0.0) {
        var e = min(in.uv.x / fu, in.uv.y / fv);
        if (kind == TRIANGLE) {
            e = min(e, (1.0 - in.uv.x - in.uv.y) / fw);
        } else {
            e = min(e, min((1.0 - in.uv.x) / fu, (1.0 - in.uv.y) / fv));
        }
        let line = 1.0 - smoothstep(width - 0.5, width + 0.5, e);
        color = mix(color, vec3<f32>(0.1, 0.1, 0.12), line);
    }

    // The template's triangles, thinner and lighter than the face's edges.
    let thin = uniforms.extra.x;
    if (thin > 0.0) {
        let t = in.corner / max(fc, vec3<f32>(1e-12));
        let d = min(t.x, min(t.y, t.z));
        let line = 1.0 - smoothstep(thin - 0.5, thin + 0.5, d);
        color = mix(color, vec3<f32>(1.0, 1.0, 1.0), 0.8 * line);
    }

    var out: Targets;
    out.color = vec4<f32>(color, 1.0);
    out.probe = vec4<f32>(value, f32(in.face), in.uv.x, in.uv.y);
    return out;
}
