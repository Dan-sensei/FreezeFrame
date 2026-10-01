# Walking people in Unreal

A frame rip holds the city's people frozen in the pose the game drew them in. In Unreal, the ones the rip caught upright walk loops through the streets with the game's own walk; the others keep their pose. It needs no manual steps: the level build makes it, and one command adds it to a level that was built before.

Tested on the Frostpunk capture `Frostpunk_20260929_115750` with UE 5.8.3 (2026-10-01). Everything here was checked against Unreal: pictures taken in the editor at fixed times match the numpy mirror of the shader frame by frame.

## What you see (Frostpunk)

- **25 people. 9 walk, 16 keep their pose:**
  - 12 lying in the snow (sick or injured);
  - 2 on crutches (`mesh_642`, `mesh_649`): one leg has no calf or foot bone (Frostpunk's amputees), and a two-legged walk can't do crutches;
  - 2 with no room to walk: `mesh_1519` stands under a ledge, `mesh_637` between a crate and a wall.
- **The walk:** heel strike, the swinging knee bent, the arms swinging opposite the legs, the body swaying 4–5 cm over the planted foot, which stays put. Adults walk 2.0 m/s, 1.1 s per cycle of two steps (Frostpunk's world is about 1.3 times life size); children 1.7 m/s, 0.92 s per cycle.
- **Where:** each one walks a closed loop. It goes along the way the game had it facing, as far as the street is clear, turns round in a half circle onto a lane 1 m beside it, and walks back. Loops are 5–38 m round. `mesh_636` has only 5.8 m of clear street, so its back and forth is obvious; better routes are the next step (see "Not done yet").
- **At time 0** each walker stands within 8–39 cm of the game's pose. The larger offsets come from people whose lane the planner turned (`mesh_325` by 60°, because the way ahead was short).
- **When:** they walk all the time in the editor viewport, like the other effects, and in Sequencer and Movie Render Queue renders, where the Level Sequence's time drives them.

## Run it

`python gtb.py unreal <capture>` builds the walkers along with the rest of the level. For a level built before 2026-10-01, open the editor on it and run:

```bash
python -m ue.live walkers Frostpunk_20260929_115750
```

It takes about 45 s and changes only walker assets and actors:

- `M_GTB_Walker` (in Shared), `MI_Look_Walker`, `T_GTB_WalkerBones` and `T_GTB_WalkerPaths`;
- the `MI_Walk_*` instances;
- the walkers' mesh assets, re-imported with their bone data;
- the walker actors: their material, their loop (custom primitive data), and the `People` folder.

It is one undo step, left unsaved: File → Save All keeps it. It prints `walker master built, compiles (1364 VS / 517 PS instructions)` and `11 walker mesh(es), 11 re-imported with bone data, 5 MI_Walk_* instance(s), loops on 11 actor(s)`. The 11 actors include the 2 that keep their pose: they use the walker material with a cycle time of 0. If a master fails to compile, every walker vanishes; the line then says `SHADER COMPILE FAILED`.

The export prints these lines (about 20 s on Frostpunk):

```
[people] 25 people: 11 walk, 2 upright on a crutch and 12 lying keep their pose
[people] walk cycle from 11 walkers: legs within 3.3 deg rms (hip x1.34 -4, knee x-1.07 +1, ankle x-1.40 +2)
[people] mesh_1538_1538: 12.9 m loop, stride 2.17 m, cycle x0.97
...
[people] 2 with no room to walk keep their pose: ['mesh_1519_1519', 'mesh_637_637']
```

The loop lines give the length of one lane, not the whole loop.

### Settings

| where | what |
|---|---|
| `look.unreal.walkers.enabled` | `false` makes everyone stand in the game's pose (`MI_Look_Walker`'s `Walk` = 0) |
| `look.unreal.walkers.period` | seconds per walk cycle (two steps) of a typical adult, default 1.1. The others scale it by the square root of their hip height, so children step faster. The stride is fixed (the planted foot must not slide), so this also sets the speed. |
| `MI_Look_Walker` | `Walk` and `WalkPeriod` come from the look; try values live there, then copy them into `look.json`. The rest (`WalkerBones`, `WalkerPaths`, `BoneRange`, `PathRange`, `Frames`, `PathSamples`) describes the capture's data; don't edit it. |

The look's surface values (albedo gain, roughness, snow and so on) reach the walkers through `MI_Look_Walker` like every other surface. The live command copies them from `MI_Look_Surface`, so hand-tuned values in the level carry over.

## Check it

Without Unreal (about 15 s):

```bash
python -m ue.walker_check Frostpunk_20260929_115750
```

For every walker it prints its loop, stride and cycle time, how fast a planted foot slips, how far its lowest point gets from the path's ground over a whole loop, and how far its pose at time 0 is from the game's. Then it lists the pairs that walk through each other. Frostpunk, expected:

- planted foot slips 0.6–1.3 cm/s median, about 2.5 p90 (walking at 1.7–2.1 m/s);
- lowest point −0.0..+0.3 cm from the ground;
- start 8–39 cm from the game's pose;
- `9 walk, 2 keep their pose, 1 pair(s) walk through each other` (`mesh_326` and `mesh_636`, who walked side by side in the game).

The ground and start numbers go through `walkers.replay`, which mirrors the shader and reads the same two textures Unreal does. Keep it in sync with `gtb_hlsl.WALKER_WPO`.

With the editor open on the capture's level, take pictures of one walker at fixed times, with the prediction drawn over them:

```bash
python -m ue.walker_check Frostpunk_20260929_115750 --capture mesh_646_646 --times 0,0.5,1,1.5
```

It places a temporary SceneCapture2D where ray casts through the rip see the walker's whole loop. It pins the time (`MPC_GTB_Time`: `SceneTime` = t, `EngineTimeWeight` = 0), takes a picture at each time, then restores both values and deletes the capture actor. `captures/<name>/unreal/walker_check/` gets each picture and an `_overlay` copy with the walker as `walkers.replay` places it, in green. The green must sit on the rendered person in every picture. Two people in the same clothes can stand close together (`mesh_326` and `mesh_636`); then the green marks which one is the walker.

In the viewport, look for:

- feet that slide;
- walkers that go through walls or each other;
- walkers that float or sink on slopes;
- shading that doesn't follow the limbs (the normals are turned to the current pose, so a swinging arm keeps its light);
- people who should walk but don't.

## How it works

### 1. Finding the people (`gtb/characters.py`)

A skinned draw has BLENDINDICES and BLENDWEIGHT in its pre-VS layout. The people are those whose bind pose is 0.8–3 m tall, with at least 16 bones that carry vertices, that the vertex shader bent (`rigid_fit` above 0.01). The rig is in the game profile's `characters` section (`profiles/frostpunk.json`):

| key | what |
|---|---|
| `bones` | bone index → [name, parent index]. The code reads the names `pelvis`, `neck`/`head`/`chest`, `spine`. |
| `legs` | [thigh, calf, foot] per leg; the first leg's heel strike is phase 0 |
| `mirror` | left/right pairs (for the cycle fit) |
| `bind_up`, `bind_forward` | the bind pose's axes in D3D object space (Frostpunk: y up, facing +z) |
| `bind_height`, `min_bones`, `min_rigid_fit`, `max_fit_error` | the filters above, and the largest skinning fit error (m, p99) accepted |

Frostpunk's rig has 24 bones with the same numbering for every person: pelvis 1, spine 2, chest 3, neck 4, head 5, clavicles 6/12, upper arms 7/13, forearms 8/14, hands 9/15, fingers 11/17, thighs 18/21, calves 19/22, feet 20/23. Bones 0, 10 and 16 carry no vertices. In the solve, +x is the person's right.

`process` keeps each skinned draw's bind pose and weights in its npz (`skin_bind`, `skin_index`, `skin_weight`). A capture processed before 2026-10-01 gets them from the rip the first time the export runs (`characters.backfill`, which prints `skin data of 263 skinned meshes read from the rip`). Without the rip, nobody walks.

### 2. Their bones

Ninja Ripper saves no bone matrices. But linear blend skinning, P = Σ w_k (R_k b + t_k), ties each posed vertex P to its bind position b through the bone transforms. With hundreds of vertices per bone, each bone's rotation and translation can be solved: Kabsch per bone (weights to the 4th power) to start, then Gauss-Newton with robust weights. All 25 Frostpunk people fit within 0.9 cm. On the way:

- **Mirror the bind pose's z first.** The bind pose is D3D, which is left-handed. Unmirrored, each bone's best fit is a reflection (determinant −1), which no rotation matches; the residuals stayed at 7–20 cm.
- **Normalise the weights.** The 8-bit weights sum to 0.996–1.000. The game skins in model space, where that's harmless. Solved in world space, 150 m from the origin, the missing 0.4% moved some vertices by 0.5 m. The solve also runs around the mesh's centroid.
- **Each person has a scale.** It's 1.3 for most Frostpunk people and 1.0 for two (`mesh_643`, `mesh_645`), measured on the bone with the most vertices bound to it alone.
- **Joints.** A joint is where parent and child move together in the captured pose: (R_p − R_b) j = t_b − t_p. One pose leaves j free along the joint's turning axis, so it is pulled towards the centroid of the vertices the two bones share. That centroid alone was up to 7 cm off at the hips: the re-posed skeleton then missed the captured mesh by up to 8 cm and the feet hung in the air. Now re-posing the captured pose reproduces the mesh within 1–1.6 cm.

### 3. The walk cycle

The 11 two-legged walkers sit at four points of the stride, a quarter cycle apart. Groups walk in step: `mesh_325`, `mesh_637` and `mesh_648` all land on the same foot at the same moment. A free two-harmonic fit, with each walker's phase solved along with it, placed them wrongly: stance and swing came out about a quarter cycle off, and the knee bent 44° while its foot was planted. So:

- **Legs:** normative gait curves (Perry/Winter style) for hip flexion, knee flexion and ankle, as functions of the cycle. Each curve's timing (each walker's phase), scale and offset is fitted to the walkers: 3.3° rms over the six leg joints. The landing knee bump was raised a little, because Frostpunk's walkers land with the knee at 21–28°.
- **Everything else** (pelvis tilt, spine, head, arms) is a first-harmonic fit. Each walker is used twice: as itself, and mirrored left to right half a cycle later.
- **The legs' twist and sideways swing** stay at the walkers' average. Fitted, they swung the planted foot sideways at up to 80 cm/s.
- The 2 amputees stay out of the fit (`Walk.phase_for` still gives them a phase). A rip with fewer than 3 walkers uses the plain template (`characters.default_walk`).

### 4. The baked cycle (`ue/walkers.py`, `bake_cycle`)

48 frames per walker, in a walking frame (x forward, ground at z 0):

- **Height:** the root rises and falls so the lowest vertex touches the ground.
- **Forward:** between frames, the root advances by how far the foot vertices that are on the ground in both frames moved back. That is the heel early in a stance and the toes at its end, so they stay put. With both feet down, the one moving back is the planted one; the other is still landing. The sum over the cycle is the stride: 2.2–2.4 m for adults, 1.6 m for children.
- **Sideways:** the same for sideways motion, which makes the 4–5 cm sway.
- **Measured:** a planted foot slips 0.6–1.3 cm/s on a straight walk. Smoothing the advance, or following the foot's centroid or contact point, slid it 25–55 cm/s. Following the contact point made it jump from heel to toe.

### 5. The loops (`plan_loop`, `clear_run`, `loop_points`)

From where the game had the person (its pelvis over the lowest point of its feet), along its heading, a lane runs as far as:

- rays at 0.35, 1.0 and 1.7 m over the ground, at the centre and ±0.3 m to the sides, hit nothing (with 0.6 m to spare);
- downward rays every 0.25 m find ground, with no step over 30 cm;
- the same holds backwards;
- at most 15 m each way.

Other headings up to ±90° are tried, at a cost of 5 cm of lane per degree. Geometry within 15 cm of the start doesn't count: the game puts people where they brush things (`mesh_1519`'s head is under a ledge). A return lane 1 m to one side must be clear for the whole length; if neither side is, the person turns about on the spot. Under 2 m of lane in all, the person keeps its pose. The ray casts use the rip's surfaces, without the walkers. People lying on the ground count as obstacles. Snow drifts count as obstacles and as ground.

The loop (lane, half circle, lane, half circle) is resampled to 128 points with the ground height under each.

### 6. In Unreal (`M_GTB_Walker`)

| data | where | what |
|---|---|---|
| per vertex | UV1, UV2 | the 4 bone indices |
| | UV3, UV4 | the 4 weights (normalised) |
| | UV5, UV6.x | the bind position: Unreal cm, standing at the origin facing +X |
| per walker | `unreal/walkers/bones.png` → `T_GTB_WalkerBones` | one row per walker and bone (24 rows each), 3 texels per frame: the rows of the 3x4 matrix (R, t) from the bind position to the walking frame. RGB = R in −1..1, A = t in ±`BoneRange` (200 cm), both stored as (v + 1) / 2. The 49th frame holds the captured pose's rotations. |
| | `unreal/walkers/paths.png` → `T_GTB_WalkerPaths` | one row per walker: 128 + 1 points (x, y, z, yaw) along its loop, relative to the actor, in ±`PathRange` per channel; the last point is the first with yaw + 2π |
| per actor | custom primitive data 0–3 (`WalkerA`) | bone row, path row, phase at time 0, cycle time (× `WalkPeriod`; 0 = keep the pose) |
| | custom primitive data 4–7 (`WalkerB`) | stride (cm per cycle), loop length (cm), start (cm along the loop), 1 |

- **The textures** are 16-bit PNGs, imported as `TC_HDR_F32` with nearest filtering, no mips and no streaming. Half floats would round the paths to 2 cm. The shader reads them with `Texture.Load`.
- **The shader** (`gtb_hlsl.WALKER_WPO`): the cycle position is frac(phase + time / cycle). Each of the 4 bones' matrices is blended between two frames, the vertex is skinned in the walking frame, then placed and turned at its point along the loop: frac((start + progress × stride) / loop length). It returns that position minus the vertex's own, as World Position Offset.
- **Normals:** the mesh keeps the captured pose's normals and tangents, so the material turns them by R_now R_captured^T. The rotation's first and last columns go to the pixel shader through two VertexInterpolators (`QX`, `QZ`). `WALKER_ROT` turns the normal map's world normal and the vertex normal the snow reads, with `tangent_space_normal` off.
- **Time** is `SceneTime + EngineTimeWeight × Time`, as in `Graph.time()`. The loop wraps, so engine time can grow without limit (unlike the smoke rise).
- **Bounds:** each walker mesh's bounds are extended over its whole loop (`walker_extent_cm`), so it isn't culled when it walks away from its spot.
- **Names:** each walker gets `MI_Walk_<its base material>` (e.g. `MI_Walk_M027`) with the base's values, so the people lying in the snow keep `MI_M027`. The `MI_M###` numbering skips the walker instances, so no other material is renumbered (`ue.live materials` points actors by name; see CLAUDE.md).

## A rip from another game

1. Find the skinned draws (BLENDINDICES in `layout_pre`) and look at their bones. Per bone, the centroid of the bind vertices weighted by w⁴ shows which part it is. Which bones share vertices (summed min(w_a, w_b)) gives the parents. The left and right halves are mirror images across the bind pose's mid-plane. This is how Frostpunk's rig was worked out.
2. Write the profile's `characters` section, with `bind_up` and `bind_forward` from the bind pose: the arms and toes point forward.
3. Run the export and read the `[people]` lines. A person skipped as `skinning fit off` has weights or indices that don't fit linear blend skinning (dual quaternions, more than 4 weights, or an index offset).
4. `python -m ue.walker_check <capture>`, then pictures in the editor with `--capture`.

## Pitfalls (what went wrong on the way)

- Every bone a reflection: the D3D bind pose wasn't mirrored.
- 0.5 m errors on a few vertices: the 8-bit weights don't sum to 1, and the solve ran in world space.
- An alternating per-bone solve (each bone Kabsch-fitted with the others held) stalled at 3–20 cm; Gauss-Newton on all bones at once converges.
- The front foot 10 cm in the air at heel strike: joint centres from shared-vertex centroids (7 cm off at the hips).
- Stance and swing a quarter cycle off, a planted knee at 44°: a free Fourier fit with free phases on walkers clustered at four stride points.
- The planted foot sliding sideways at 80 cm/s: the fitted twist and sideways swing of the legs.
- Feet sliding 25–55 cm/s: a smoothed root advance, or one that followed the foot's centroid or contact point.
- Two walkers stuck in place: one stood touching a ledge (fixed by ignoring geometry within 15 cm of the start), the other is boxed in.
- Path planning took minutes: rays tested against every triangle within 18 m. Each cast now uses a thin box around its line, so the export spends about 15 s on it.
- The editor capture camera inside a wall: place it by ray casts that see the loop.
- `unreal.KismetMaterialLibrary` doesn't exist in Python; it is `unreal.MaterialLibrary`.

## Not done yet

- **Routes through the city.** The lanes are short and straight. Next: a walkable map of the whole city from the rip, a grid of about 0.5 m cells with the same ground and body-height tests as the lanes. Then longer routes on it (closed circuits or round trips of tens of metres along the streets, with gentle curves), preferring roads if the road material can be told apart. The shader takes any closed path, so only `ue/walkers.py` changes.
- **Walkers avoiding each other:** `mesh_326` and `mesh_636` pass through each other (`walker_check` lists such pairs).
- **Crutch walkers** (`mesh_642`, `mesh_649`) need their own cycle.
- **People at work:** every upright person on Frostpunk was walking. Someone standing still would walk too.
- **Blender** has no walking.

## Files

| path | what |
|---|---|
| `gtb/characters.py` | Skin data (`skin_arrays`, `backfill`), the rig, the bone solve (`Person`), the walk cycle (`fit_walk`, `Walk`) |
| `ue/walkers.py` | Finding the walkers, the baked cycle, the loops, the textures and plan entries; `replay` mirrors the shader |
| `ue/walker_check.py` | The checks above |
| `ue/editor/gtb_hlsl.py` | `WALKER_WPO`, `WALKER_ROT` |
| `ue/editor/gtb_materials.py` | `build_walker` (`M_GTB_Walker`: the surface master plus the walker parts) |
| `ue/editor/gtb_ue.py` | Imports the textures, sets `MI_Look_Walker` and each walker's custom primitive data and bounds |
| `ue/live.py` | `walkers`: the same, into an open editor |
| `ue/export.py` | Walker meshes get the bone UVs and `MI_Walk_*`; the plan's `walkers` section |
| `profiles/frostpunk.json` | The `characters` rig |
| `tests/selftest.py` | A synthetic person: bones solved, planted foot, closed loop |
