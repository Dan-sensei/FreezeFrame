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
# UV1.xy/UV2.x: vertex -> puff centre; UV3: corner along the game camera's right/up;
# UV6.x: random phase per puff. In the Level Sequence (and renders) puffs drift up
# with the sequence time like Blender. In the editor / Play (engine time, W = 1,
# which grows without end) they instead rise and fade in a loop of LoopSeconds, so
# the viewport is alive but the smoke stays where the game drew it.
SPRITE_WPO = r"""
float3 off = float3(0.0, 0.0, Rise * TSeq);
float grow = 1.0;
if (LoopSeconds > 0.0 && W > 0.5)
{
    float u = frac(TAll / LoopSeconds + Phase.x);
    off.z = Rise * LoopSeconds * u;
    grow = 1.0 + 0.4 * u;
}
if (Billboard > 0.5)
    off += float3(D1.x, D1.y, D2.x) + (Corner.x * CamR + Corner.y * CamU) * grow;
return off;
"""

# Fade for the editor loop (see SPRITE_WPO): in at the bottom, out at the top.
LOOP_FADE = r"""
float fade = 1.0;
if (LoopSeconds > 0.0 && W > 0.5)
{
    float lu = frac(TAll / LoopSeconds + Phase.x);
    fade = saturate(lu / 0.15) * saturate((1.0 - lu) / 0.35);
}
"""

# build_sprite_material (smoke): flipbook alpha x vertex alpha x opacity; the
# sheet's RGB is a tangent-space normal along the sprite's texture axes.
SMOKE = LOOP_FADE + r"""
float4 a = Texture2DSample(Atlas, AtlasSampler, UV);
Alpha = min(a.a * VColorA * Opacity * fade, 1.0);
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

FIRE = LOOP_FADE + r"""
float4 a = Texture2DSample(Atlas, AtlasSampler, UV);
Alpha = min(a.a * VColorA * Opacity * fade, 1.0);
return a.rgb * FireStrength;
"""

# Smoke column (no Blender counterpart in the looks: Blender's volumetric plume is
# off in the Frostpunk presets). Puffs leave the vent at their phase, rise the
# game's column plus Height over Period, widen by Grow per cm, drift with the wind
# and spin. UV1 = (phase, angle), UV2 = jitter, UV3 = corner (+-0.5).
PLUME_WPO = r"""
float u = frac(T / Period + P1.x);
float h = (Column + Height) * u;
float r = Radius * (1.0 + Grow * h);
float2 cs = float2(cos(P1.y), sin(P1.y));
float2 c = float2(Corner.x * cs.x - Corner.y * cs.y, Corner.x * cs.y + Corner.y * cs.x) * r * 2.4;
float3 centre = Base + float3(P2.xy * r * 0.45 + Drift.xy * h, h);
return centre + c.x * CamR + c.y * CamU - WorldPos;
"""

# Hanging cloth (banners): pinned along the top of the object's bounds, free at
# the bottom. Ripples travel down the cloth across its plane (vertex normal), and
# the whole strip swings downwind with slow gusts. Ripple and Sway are fractions
# of the cloth's length; the phase is per object so neighbours move out of step.
# Room (custom primitive data 0-11, from ue/export.cloth_room): the banner's front
# direction (Room.xy), Room.z = 1 when the bottom edge is held too (inside a
# beam: the motion fades out towards it), Room.w = 1 when the side
# edges are held (no sideways swing; the motion across the plane fades to 0 at
# the edges, see export.cloth_taper), and BudA/BudB = 8 budgets,
# cm per unit motion weight w, for directions k * 45 degrees from front towards
# across = (-front.y, front.x), each covering its whole 45-degree sector.
# Every motion term is (constant per banner) x w x gust, so the motion is fitted
# to the budgets once per banner, at the strongest gust: the ripple across the
# plane shrinks (and billows to the free side when a wall is close), the wind
# swing is limited per side, and a last uniform scale keeps the extremes inside
# the diagonal budgets. Clamping each vertex per frame instead made them snap.
# Twins (front/back meshes, opposite vertex normals) share the front direction.
# Mirrored in ue/cloth_check.cloth_offsets; keep w in sync with export.cloth_weight.
# Unreal has no Blender counterpart yet (look.unreal.cloth).
CLOTH_WPO = r"""
float len = max(BMax.z - BMin.z, 1.0);
float halfw = 0.5 * length(BMax.xy - BMin.xy);
float sides = Room.w;
float d = saturate((BMax.z - Local.z) / len);
float held = Room.z;
float w = d * d * lerp(1.0, saturate((1.0 - d) / 0.35), held);
float phase = frac(sin(dot(ObjPos.xy, float2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
float ws = length(Wind.xy);
float2 wd = ws > 1e-3 ? Wind.xy / ws : float2(1.0, 0.0);
float smax = min(ws / 250.0, 1.5);
float gust = 0.7 + 0.3 * sin(T * 0.45 + phase) * sin(T * 0.23 + phase * 1.3);
float2 n = normalize(Normal.xy + 1e-4);
bool hasRoom = dot(Room.xy, Room.xy) > 0.25;
if (hasRoom)
    n = Room.xy;
else if (n.x < 0.0 || (n.x == 0.0 && n.y < 0.0))
    n = -n;
float2 across = float2(-n.y, n.x);
float halfA = 0.5 * dot(abs(across), BMax.xy - BMin.xy);
float xa = dot(Local.xy - 0.5 * (BMin.xy + BMax.xy), across);
float wn = w * lerp(1.0, saturate((1.0 - abs(xa) / max(halfA, 1e-3)) / 0.4), sides);
float bud[8] = {BudA.x, BudA.y, BudA.z, BudA.w, BudB.x, BudB.y, BudB.z, BudB.w};
if (!hasRoom)
    for (int i = 0; i < 8; i++) bud[i] = 1e6;
// WaveLength is a fraction of the cloth's length: the game's banners are all the
// same 4x9 grid, so a fixed wavelength aliased into zigzags on the long ones.
float sp = d * 6.2831853 / max(WaveLength, 0.3);
float wt = T * Speed * 6.2831853;
float travel = sp - wt + phase;
float ripple = sin(travel) + 0.35 * sin(0.6 * sp - 1.7 * wt + phase * 2.1);
// Twist: the side edges turn about the centre line together (same timing across
// the width; a phase that varied across it made the edges fight each other).
float twist = 0.12 * dot(Local.xy, across) * sin(0.7 * travel + 1.3);
// Fit at the strongest gust (per unit w): ripple amplitude a around centre C
// across the plane, swing SA along the wall.
float A = Ripple * len * 1.35 + 0.12 * halfw;
float rn = (Ripple * len * ripple + twist) / max(A, 1e-3);    // -1..1
float a = min(A * smax, 0.5 * (bud[0] + bud[4]));
// No swing when the bottom is held too: a banner fixed at both ends can't lean
// with the wind, and the swing only bowed its middle.
float swing = Sway * len * smax * (1.0 - held);
float C = clamp(swing * dot(wd, n), a - bud[4], bud[0] - a);
float SA = clamp(swing * dot(wd, across), -bud[6], bud[2]) * (1.0 - sides);
float f = 1.0;
for (int j = -1; j <= 1; j++)
{
    float2 v = float2(C + j * a, SA);
    float r = length(v);
    if (r > 1e-3)
    {
        float k = frac(atan2(v.y, v.x) / 6.2831853) * 8.0;
        int k0 = (int)floor(k) % 8;
        f = min(f, lerp(bud[k0], bud[(k0 + 1) % 8], k - floor(k)) / r);
    }
}
float3 off = float3((n * (C + a * rn) * wn + across * SA * w) * f * gust, 0.0);
off.z = dot(off.xy, off.xy) / (2.0 * max(d * len, 1.0)) * (1.0 - held);
return off;
"""

PLUME = r"""
float4 a = Texture2DSample(Atlas, AtlasSampler, UV);
float u = frac(T / Period + P1.x);
float fade = saturate(u / 0.06) * saturate((1.0 - u) / 0.55);
Alpha = min(a.a * VColorA * Opacity * fade, 1.0);
float2 cs = float2(cos(P1.y), sin(P1.y));
float xs = a.r * 2.0 - 1.0;
float ys = a.g * 2.0 - 1.0;
float zs = sqrt(max(1.0 - xs * xs - ys * ys, 0.0));
float3 nf = -CamF;
float3 tw = cs.x * CamR + cs.y * CamU;
float3 bw = -cs.y * CamR + cs.x * CamU;
WorldN = normalize(lerp(nf, normalize(xs * tw + ys * bw + zs * nf), NormalStrength));
// Furnace light on the smoke just above the vent.
Emis = SmokeColor.rgb * Ambient + GlowColor.rgb * Glow * pow(saturate(1.0 - u * 5.0), 2.0);
return SmokeColor.rgb;
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
