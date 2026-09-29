"""HLSL for the master materials' Custom nodes.

Each snippet mirrors a node tree in blender/gtb_scene.py so both engines shade
the same way; comments name the Blender counterpart. Unreal world space is
Blender's with Y negated and centimetres (see ue/export.py).
"""

# Cycles' Perlin noise (intern/cycles/util/noise.h: hash_uint3, grad3, perlin_3d,
# noise_fbm) so the snow patches land exactly where Blender puts them.
NOISE = r"""
struct GTBNoise
{
    uint Rot(uint x, uint k) { return (x << k) | (x >> (32u - k)); }
    uint Hash(uint kx, uint ky, uint kz)
    {
        uint a = 0xdeadbeefu + (3u << 2) + 13u;
        uint b = a;
        uint c = a;
        c += kz; b += ky; a += kx;
        c ^= b; c -= Rot(b, 14u);
        a ^= c; a -= Rot(c, 11u);
        b ^= a; b -= Rot(a, 25u);
        c ^= b; c -= Rot(b, 16u);
        a ^= c; a -= Rot(c, 4u);
        b ^= a; b -= Rot(a, 14u);
        c ^= b; c -= Rot(b, 24u);
        return c;
    }
    float Grad(uint h, float x, float y, float z)
    {
        h &= 15u;
        float u = h < 8u ? x : y;
        float vt = (h == 12u || h == 14u) ? x : z;
        float v = h < 4u ? y : vt;
        return ((h & 1u) != 0u ? -u : u) + ((h & 2u) != 0u ? -v : v);
    }
    float Fade(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
    float Perlin(float3 p)
    {
        float3 fl = floor(p);
        float3 f = p - fl;
        int3 i = int3(fl);
        uint X = asuint(i.x);
        uint Y = asuint(i.y);
        uint Z = asuint(i.z);
        float u = Fade(f.x);
        float v = Fade(f.y);
        float w = Fade(f.z);
        float v0 = Grad(Hash(X, Y, Z), f.x, f.y, f.z);
        float v1 = Grad(Hash(X + 1u, Y, Z), f.x - 1.0, f.y, f.z);
        float v2 = Grad(Hash(X, Y + 1u, Z), f.x, f.y - 1.0, f.z);
        float v3 = Grad(Hash(X + 1u, Y + 1u, Z), f.x - 1.0, f.y - 1.0, f.z);
        float v4 = Grad(Hash(X, Y, Z + 1u), f.x, f.y, f.z - 1.0);
        float v5 = Grad(Hash(X + 1u, Y, Z + 1u), f.x - 1.0, f.y, f.z - 1.0);
        float v6 = Grad(Hash(X, Y + 1u, Z + 1u), f.x, f.y - 1.0, f.z - 1.0);
        float v7 = Grad(Hash(X + 1u, Y + 1u, Z + 1u), f.x - 1.0, f.y - 1.0, f.z - 1.0);
        float x1 = 1.0 - u;
        float y1 = 1.0 - v;
        float z1 = 1.0 - w;
        return z1 * (y1 * (v0 * x1 + v1 * u) + v * (v2 * x1 + v3 * u)) +
               w * (y1 * (v4 * x1 + v5 * u) + v * (v6 * x1 + v7 * u));
    }
    // Noise Texture node: fBM, normalized, roughness 0.5, lacunarity 2, integer detail.
    float Fbm(float3 p, int detail)
    {
        float fscale = 1.0;
        float amp = 1.0;
        float maxamp = 0.0;
        float sum = 0.0;
        [loop] for (int i = 0; i <= detail; i++)
        {
            sum += 0.9820 * Perlin(fscale * p) * amp;
            maxamp += amp;
            amp *= 0.5;
            fscale *= 2.0;
        }
        return 0.5 * sum / maxamp + 0.5;
    }
};
"""

# build_material + add_surface_snow. Outputs: return = base colour; Rough, Metal,
# Spec, NormalTS (tangent space), Emis, Alpha.
SURFACE = NOISE + r"""
GTBNoise N;
float3 base = BaseColor.rgb;
float alpha = 1.0;
if (UseAlbedo > 0.5)
{
    float4 a = Texture2DSample(AlbedoTex, AlbedoTexSampler, UV);
    float3 c = a.rgb;
    // Hue/Saturation/Value node (hue kept): saturation and value scaled in HSV.
    float mx = max(c.r, max(c.g, c.b));
    float mn = min(c.r, min(c.g, c.b));
    float d = mx - mn;
    if (mx > 0.0 && d > 1e-8)
    {
        float s2 = saturate(d / mx * Saturation);
        c = mx * (1.0 - s2 * (mx - c) / d);
    }
    c *= AlbedoGain;
    if (MultiplyVertexColor > 0.5)
        c *= VColor.rgb;
    base = c;
    alpha = a.a;          // a cut-out mask, never blended opacity
}
else if (IsGround > 0.5)
{
    base = lerp(base, GroundColor.rgb, GroundColorWeight);
}

float rough = Roughness;
float metal = 0.0;
float spec = Specular;
float3 nts = float3(0.0, 0.0, 1.0);
if (UseNormal > 0.5)
{
    float4 t = Texture2DSample(NormalTex, NormalTexSampler, UV);
    // Two-channel maps (RG, or AG with X in alpha) get Z rebuilt.
    float x = NormalAG > 0.5 ? t.a : t.r;
    float y = NormalFlipGreen > 0.5 ? 1.0 - t.g : t.g;
    float xs = x * 2.0 - 1.0;
    float ys = y * 2.0 - 1.0;
    float zs = NormalRebuildZ > 0.5 ? sqrt(max(1.0 - xs * xs - ys * ys, 0.0)) : t.b * 2.0 - 1.0;
    float3 n = normalize(float3(xs, ys, zs));
    nts = normalize(lerp(float3(0.0, 0.0, 1.0), n, max(NormalStrength, 0.0)));
    // Engines that pack the normal into A+G keep masks in R and B.
    if (dot(NormalRoughMask, 1.0) > 0.5)
    {
        float r = dot(t, NormalRoughMask);
        rough = NormalRoughInvert > 0.5 ? 1.0 - r : r;
    }
    if (dot(NormalMetalMask, 1.0) > 0.5)
    {
        float m = dot(t, NormalMetalMask);
        metal = NormalMetalInvert > 0.5 ? 1.0 - m : m;
    }
}
if (UseGray > 0.5)
{
    float g = Texture2DSample(GrayTex, GrayTexSampler, UV).r;
    g = GrayIsGloss > 0.5 ? 1.0 - g : g;
    rough = lerp(Roughness, g, RoughnessFromGray);
}
if (UsePacked > 0.5)
{
    float4 p = Texture2DSample(PackedTex, PackedTexSampler, UV);
    if (dot(PackedRoughMask, 1.0) > 0.5)
    {
        float r = dot(p, PackedRoughMask);
        rough = PackedRoughInvert > 0.5 ? 1.0 - r : r;
    }
    if (dot(PackedMetalMask, 1.0) > 0.5)
    {
        float m = dot(p, PackedMetalMask);
        metal = PackedMetalInvert > 0.5 ? 1.0 - m : m;
    }
}
float3 emis = 0.0;
if (UseEmissive > 0.5)
    emis = Texture2DSample(EmissiveTex, EmissiveTexSampler, UV).rgb * EmissionStrength;

if (UseSnow > 0.5)
{
    // Snow on up-facing surfaces (the game adds it in-shader). Blender's
    // Geometry normal faces the viewer on back faces.
    float3 nw = normalize(VNormal);
    nw = dot(nw, CamVec) < 0.0 ? -nw : nw;
    float lo = SnowThreshold - 0.12;
    float hi = SnowThreshold + 0.12;
    float k = saturate((nw.z - lo) / (hi - lo));
    float ramp = k * k * (3.0 - 2.0 * k);
    float3 pb = float3(WorldPos.x, -WorldPos.y, WorldPos.z) * 0.01;   // Blender world, metres
    float nz = N.Fbm(pb * 0.6, 8);
    float patch = saturate((nz - (1.0 - SnowCover)) / 0.2);
    float f = ramp * patch;
    base = lerp(base, float3(0.86, 0.9, 0.97), f);
    rough = lerp(rough, 0.55, f);
}
Rough = rough;
Metal = metal;
Spec = spec;
NormalTS = nts;
Emis = emis;
Alpha = alpha;
return base;
"""

# build_snowmesh_material: snow colour, the shared snow texture's RG as a normal.
SNOWDRIFT = r"""
float4 t = Texture2DSample(SnowTex, SnowTexSampler, UV);
float y = NormalFlipGreen > 0.5 ? 1.0 - t.g : t.g;
float3 n = normalize(float3(t.r * 2.0 - 1.0, y * 2.0 - 1.0, 1.0));
NormalTS = normalize(lerp(float3(0.0, 0.0, 1.0), n, max(NormalStrength, 0.0)));
return DriftColor.rgb;
"""

# billboard_node_group: re-face each puff to the rendering camera and drift up.
# UV1.xy/UV2.x: vertex -> puff centre; UV3: corner along the game camera's right/up.
SPRITE_WPO = r"""
float3 off = float3(0.0, 0.0, Rise * T);
if (Billboard > 0.5)
    off += float3(D1.x, D1.y, D2.x) + Corner.x * CamR + Corner.y * CamU;
return off;
"""

# build_sprite_material (smoke): flipbook alpha x vertex alpha x opacity; the
# sheet's RGB is a tangent-space normal along the sprite's texture axes.
SMOKE = r"""
float4 a = Texture2DSample(Atlas, AtlasSampler, UV);
Alpha = min(a.a * VColorA * Opacity, 1.0);
float xs = a.r * 2.0 - 1.0;
float ys = a.g * 2.0 - 1.0;
float zs = sqrt(max(1.0 - xs * xs - ys * ys, 0.0));
float3 nf = -CamF;
float3 tw = normalize(Tan.x * CamR + Tan.y * CamU + 1e-6);
float3 bw = normalize(Bit.x * CamR + Bit.y * CamU + 1e-6);
float3 n = normalize(xs * tw + ys * bw + zs * nf);
WorldN = normalize(lerp(nf, n, NormalStrength));
Emis = SmokeColor.rgb * Ambient;
return SmokeColor.rgb;
"""

FIRE = r"""
float4 a = Texture2DSample(Atlas, AtlasSampler, UV);
Alpha = min(a.a * VColorA * Opacity, 1.0);
return a.rgb * FireStrength;
"""

# snow_node_group: flakes as a pure function of time, wrapped in a box that
# follows the camera (or the capture's snow volume), faced to the camera.
# UV1 = random start xy, UV2 = (random start z, size factor), UV3 = (index, phase).
SNOW_WPO = r"""
float3 rnd = float3(R1.x, R1.y, R2.x);
bool follow = Follow > 0.5;
float3 lo = follow ? CamPos - Extent.xyz : BoxMin.xyz;
float3 hi = follow ? CamPos + Extent.xyz : BoxMax.xyz;
float3 slo = follow ? -Extent.xyz : BoxMin.xyz;
float3 shi = follow ? Extent.xyz : BoxMax.xyz;
float3 p = lerp(slo, shi, rnd);
float ph = R3.y * 6.2831853;
p.xy += Wind.xy * T + float2(sin(T * 0.9 + ph), sin(T * 0.7 + ph * 1.7)) * 0.15 * Flutter;
p.z -= Fall * T;
float3 range = max(hi - lo, 1e-3);
p -= range * floor((p - lo) / range);
float r = Size * R2.y * (R3.x < Count ? 1.0 : 0.0);
float2 c = (Corner - 0.5) * 2.0 * r;
return p + c.x * CamR - c.y * CamU - WorldPos;
"""

# snow_material: soft round flake. Flakes right in front of the lens would
# fill the frame with a blurry disc, so they fade out within NearFade.
SNOW_ALPHA = r"""
float fall = saturate((0.5 - length(UV - 0.5)) / 0.5);
float near = saturate(length(WorldPos - CamPos) / max(NearFade, 1.0) - 0.5);
return pow(fall, 1.6) * Opacity * near;
"""

# Blender's compositor grade + view transform, baked by ue/blender_lut.py.
# Runs before Unreal's tonemapper, which the look sets to a neutral pass-through
# (Filmic with tone curve 0): so output the display value decoded to linear and
# undo the exposure the tonemapper applies. Unreal then encodes for whatever the
# output is (sRGB viewport, Movie Render Queue's linear capture, ...). Scene colour
# here is exposure-free (pre-exposure is handled by the engine).
TONEMAP = r"""
float3 c = Scene.rgb * Scale;
float n = Size;
float3 t = saturate((log2(max(c, 1e-12)) - Lo) / (Hi - Lo)) * (n - 1.0);
float b0 = floor(t.b);
float b1 = min(b0 + 1.0, n - 1.0);
float fb = t.b - b0;
float v = (t.g + 0.5) / n;
float3 A = Texture2DSampleLevel(Lut, LutSampler, float2((b0 * n + t.r + 0.5) / (n * n), v), 0).rgb;
float3 B = Texture2DSampleLevel(Lut, LutSampler, float2((b1 * n + t.r + 0.5) / (n * n), v), 0).rgb;
float3 d = saturate(lerp(A, B, fb));
float3 lin = lerp(pow((d + 0.055) / 1.055, 2.4), d / 12.92, step(d, 0.04045));
return lin / max(Exposure, 1e-8);
"""
