# Walking people in Unreal

A frame rip holds the city's people frozen in the pose the game drew them in. In Unreal, the ones the rip caught upright walk up and down the city's streets with the game's own walk; the others keep their pose. It needs no manual steps: the level build makes it, and one command adds it to a level that was built before.

Tested on the Frostpunk capture `Frostpunk_20260929_115750` with UE 5.8.3 (2026-10-01). Everything here was checked against Unreal: pictures taken in the editor at fixed times match the numpy mirror of the shader frame by frame.

## What you see (Frostpunk)

- **25 people. 9 walk, 16 keep their pose:**
  - 8 walk up and down a street of the city, 60–115 m each way, 82–100% of it on the game's roads;
  - 1 (`mesh_652`, a child) stands 3.8 m from open street and walks a 2.6 m lane on the spot it has;
  - 12 lie in the snow (sick or injured) and keep their pose;
  - 2 on crutches (`mesh_642`, `mesh_649`) keep theirs: one leg has no calf or foot bone (Frostpunk's amputees), and a two-legged walk can't do crutches;
  - 2 have no way out and keep theirs: `mesh_1519` stands under a ledge, `mesh_637` between a crate and a wall.
- **The walk:** heel strike, the swinging knee bent, the arms swinging opposite the legs. The body moves at a steady speed, sways 2.5 cm each way over the planted foot and rises and falls 4–8 cm; the planted foot stays put (it slips about 1 cm/s) and the swinging foot clears the ground by 5 cm mid-swing. Adults walk 1.9–2.0 m/s, 1.1 s per cycle of two steps (Frostpunk's world is about 1.3 times life size); children 1.6–1.7 m/s, 0.92 s per cycle.
- **Where:** each one walks on along the street the game had it on, in the direction it was facing, to a point up to 60 m ahead, turns round, walks back past its start to a point as far behind, turns round again, and so on. The routes follow the game's ring-and-spoke roads round corners and junctions. Everyone keeps 0.75 m to the right of their way, so the way up and the way back are two lanes and people coming the other way pass instead of walking through each other.
- **At time 0** each walker stands within 8–62 cm of the game's pose, walking the way the game had it facing.
- **When:** they walk all the time in the editor viewport, like the other effects, and in Sequencer and Movie Render Queue renders, where the Level Sequence's time drives them.

## Run it

`python gtb.py unreal <capture>` builds the walkers along with the rest of the level. For a level built before 2026-10-01, open the editor on it and run:

```bash
python -m ue.live walkers Frostpunk_20260929_115750
```

It takes about 2 minutes (most of it is the export, which plans the routes) and changes only walker assets and actors:

- `M_GTB_Walker` (in Shared), `MI_Look_Walker`, `T_GTB_WalkerBones` and `T_GTB_WalkerPaths`;
- the `MI_Walk_*` instances;
- the walkers' mesh assets, re-imported with their bone data;
- the walker actors: their material, their loop (custom primitive data), and the `People` folder.

It is one undo step, left unsaved: File → Save All keeps it. It prints `walker master built, compiles (1364 VS / 517 PS instructions)` and `11 walker mesh(es), 11 re-imported with bone data, 5 MI_Walk_* instance(s), loops on 11 actor(s)` (0 re-imported when the meshes already have their bone data). The 11 actors include the 2 that keep their pose: they use the walker material with a cycle time of 0. If a master fails to compile, every walker vanishes; the line then says `SHADER COMPILE FAILED`.

The export prints these lines (about 50 s of the export on Frostpunk; the walkable map takes 16–21 s of it):

```
[people] 25 people: 11 walk, 2 upright on a crutch and 12 lying keep their pose
[people] walk cycle from 11 walkers: legs within 3.3 deg rms (hip x1.34 -4, knee x-1.07 +1, ankle x-1.40 +2)
[people] roads: MI_M000 (267 meshes; 5 of 11 people stand on it)
[people] walkable map: 205191 floors (122705 in the main network, 16042 on roads), 16 s
[people] mesh_1538_1538: 121 m street up and back (96% on roads), stride 2.17 m, cycle x0.97
...
[people] mesh_652_652: 2.6 m lane, stride 1.57 m, cycle x0.84
[people] 2 with no room to walk keep their pose: ['mesh_1519_1519', 'mesh_637_637']
```

A street line gives the street's length (the walker covers it twice per round); a lane line is the fallback for someone who can't reach the streets.

### Settings

| where | what |
|---|---|
| `look.unreal.walkers.enabled` | `false` makes everyone stand in the game's pose (`MI_Look_Walker`'s `Walk` = 0) |
| `look.unreal.walkers.period` | seconds per walk cycle (two steps) of a typical adult, default 1.1. The others scale it by the square root of their hip height, so children step faster. The stride is fixed (the planted foot must not slide), so this also sets the speed. |
| `MI_Look_Walker` | `Walk` and `WalkPeriod` come from the look; try values live there, then copy them into `look.json`. The rest (`WalkerBones`, `WalkerPaths`, `BoneRange`, `PathRange`, `Frames`, `PathSamples`) describes the capture's data; don't edit it. |

The look's surface values (albedo gain, roughness, snow and so on) reach the walkers through `MI_Look_Walker` like every other surface. The live command copies them from `MI_Look_Surface`, so hand-tuned values in the level carry over.

## Check it

Without Unreal (about 35 s):

```bash
python -m ue.walker_check Frostpunk_20260929_115750
```

For every walker it prints its loop, stride and cycle time, the body's forward speed over the cycle (as a share of the average: it must stay at 100–100%, or the walker surges in every step) and its rise and fall, how fast a planted foot slips (the mean motion of the vertices on the ground of the foot carrying the person), how far its lowest point gets from the path's ground over a whole loop, and how far its pose at time 0 is from the game's. Then the routes that run into geometry (rays from each path point to the next at 0.5, 1.0 and 1.7 m) and the pairs that walk through each other. It also draws `captures/<name>/unreal/walker_check/routes.png`: a top view with the roads light, everything else that's built dark, open ground white, each route in its own colour and each walker's start circled. Frostpunk, expected:

- body speed 100–100% of the average, bob 4–8 cm;
- planted foot slips about 1 cm/s median, 2.5–8 p90 (walking at 1.6–2.0 m/s);
- lowest point −0.0..+0.3 cm from the ground;
- start 7–68 cm from the game's pose;
- no route runs into geometry (an earlier version brushed a road tile's snow bank on `mesh_648`'s route);
- `9 walk, 2 keep their pose, 2 pair(s) walk through each other`: `mesh_646` crosses the routes of `mesh_326` and `mesh_636` at a junction just as they pass (closest 9–14 cm, for a moment). Walkers on the same street never meet head-on (two lanes), but they don't wait for each other at crossings.

Also check how smooth the routes are: on a straight street the walker shouldn't turn. Measured from the path texture's yaw, the turn rate is about 0.1 rad/m median and 0.3–0.5 rad/m p90; the highest values (2–6 rad/m) are the half-circle turns at the street ends. Strong turns that flip side (left then right, over 0.3 rad/m) happen 0–1 times per route; when the lane offset was switched point by point, there were 3–19, and the walkers zigzagged.

The ground and start numbers go through `walkers.replay`, which mirrors the shader and reads the same two textures Unreal does. Keep it in sync with `gtb_hlsl.WALKER_WPO`.

With the editor open on the capture's level, take pictures of one walker at fixed times, with the prediction drawn over them:

```bash
python -m ue.walker_check Frostpunk_20260929_115750 --capture mesh_646_646 --times 0,0.5,1,1.5
```

It places a temporary SceneCapture2D where ray casts through the rip see the walker's whole loop. A street route is too long for one camera: add `--follow` and pick times seconds apart (`--times 0,8,16,24,32`), and it places a camera beside the walker for each picture. It pins the time (`MPC_GTB_Time`: `SceneTime` = t, `EngineTimeWeight` = 0), takes a picture at each time, then restores both values and deletes the capture actor. `captures/<name>/unreal/walker_check/` gets each picture and an `_overlay` copy with the walker as `walkers.replay` places it, in green. The green must sit on the rendered person in every picture. Two people in the same clothes can stand close together (`mesh_326` and `mesh_636`); then the green marks which one is the walker.

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

96 frames per walker, in a walking frame (x forward, ground at z 0). The body moves at a steady speed: the shader carries it along its path, and the bones add only its sway and its rise and fall. The feet are kept planted by re-timing the legs and by foot locks:

- **A planted foot** (`_planted`) is one whose lowest point is within 4 mm of the ground in two frames running. Its vertices within 1.5 cm of the ground then carry the person. A looser test (1.5 cm) took the swinging foot right after toe-off for a planted one: the game's toe stays that close for a while.
- **Re-timing** (`leg_timing`): the game's cycle (and the gait curves fitted to it) moves a planted foot back at 1.4–4.1 m per cycle within one step, slowly mid-step and fast before it lifts. While each foot is down, its leg's poses play faster or slower, so the foot moves back at an even pace (afterwards within about 2% between the 10th and 90th percentile). It eases in and out over 8 of 192 frames at heel strike and toe-off. The poses are the game's, only their timing within the step changes, and the steps per second stay the same.
- **Stride:** per leg, the even pace a planted foot moves back at; averaged over the two legs, times one cycle: 2.1–2.3 m for adults and 1.5 m for children.
- **Rise and fall:** the height that puts the lowest vertex on the ground, smoothed to 4 harmonics.
- **Sway:** 2.5 cm each way, over the first leg at its mid-stance (phase 0.3).
- **Foot locks** (`_foot_locks`): per leg and frame, how far its foot must move so that the vertices carrying the person stay put. While the foot is up, the lock eases back to 0 (×0.93 per frame of 192), and the foot is lifted so that its lowest point clears the ground by 5 cm × sin(π × progress through the swing). The game's own swing passes only 1–5 cm over the ground.
- **IK** (`leg_ik`): two bones, hip to knee to ankle. The hip stays, the knee bends in the plane it had, the foot keeps its orientation, and the ankle reaches the locked spot. Where a planted foot is out of reach, it first rolls about the part of it on the ground, its toes at push-off or its heel at landing (`foot_pivot`, up to 60°). Only then is the body lowered there (smoothed).
- **Measured** (`walker_check`): the body speed is the same all through the cycle, planted feet slip about 1 cm/s median, nothing floats or sinks more than 2 mm, and the swinging foot clears the ground by 5–9 cm.

### 5. Routes through the city (`ue/routes.py`)

**The walkable map** (`WalkMap`), over the walkers' part of the scene plus 65 m around (Frostpunk: 205,191 floors in 16–21 s):

- Points 0.2 m apart on every solid surface (the rip's surfaces without the walkers and hidden effects; banners, lying people and snow drifts included) fall into 0.25 m boxes on a 0.5 m grid of columns.
- In each column, a run of filled boxes whose top faces up (normal z > 0.5) with 1.9 m free above it is a floor, at the mean height of the up-facing points in its top box. How thick the run is doesn't matter.
- Floors of neighbouring columns (8 directions) within 35 cm of each other are linked. A wall, post or crate top doesn't link to the ground beside it, so it blocks.
- A floor whose 8 neighbours are all linked keeps 0.5 m from anything solid (`interior`); with its neighbours' neighbours too, 0.75 m (`inner`). Tight floors (next to something) cost 6 times as much, interior ones 2 times, inner ones 1. They stay usable: forbidding them broke Frostpunk's city into hundreds of islands at the gaps between buildings and road-side stands.
- The main network is the largest linked set (Frostpunk: 122,705 floors, the whole city and the open snow around it).

**Roads:** the textured surface the most walkers stand on (at least 3 of them; the bare terrain has no UVs and doesn't count). On Frostpunk that is `MI_M000`: 267 tiles, the snow paths of the ring-and-spoke streets, which 5 of the 11 walkers stand on. Every column under a road triangle is road. Walking off the road costs 2.5 times as much.

**A route** (`plan_route`):

- **Start:** the floor nearest the person (up to 1.5 m away, not a tight one) that it can step to in a straight line over floors, not counting its own 0.3 m. Without one, it gets a lane (below). That is `mesh_652`, 3.8 m from the main network.
- **Ahead:** from the start, the cheapest ways (Dijkstra). The far end ahead is a road floor 25–60 m away, the one whose way leaves the start most along the person's heading, the further the better. If nothing lies ahead, the person turns round.
- **Behind:** the far end behind is up to 60 m away, on a road if there are any, leaving the start the opposite way from the way ahead and avoiding its cells.
- **Shape:** the route is the street from the far end behind, through the start, to the far end ahead, walked there and back. So the person starts in mid-street, walking on the way the game had it.

**The line** (`route_points`):

1. **Straighten.** From each kept floor, jump to the furthest one up to 10 m on that a person can walk to in a straight line over interior floors, staying on road columns where the street is road. That gives runs of straight lines instead of the grid's 8-direction staircase.
2. **Round the corners.** Smooth over ±1.5 m, then ±0.75 m, keeping a smoothed point only where it (and the step to the next) stays on interior floors.
3. **Room to turn.** Shorten an end of the street (up to 3 m, never past the start) until a half circle of 0.75 m fits there.
4. **Build the loop.** Up the street on the right-hand lane (0.75 m right of the way; directions smoothed over ±1 m), a half circle round the far end, back on the other lane, a half circle round the near end.
5. **Lane width.** Work out how much of the lane offset each point may take: the most of 0, ⅓, ⅔ or all of it that stays on interior floors. Then take the least within ±1.5 m and average over ±1.5 m, so a lane narrows gradually where the street does. Where a point or a step still leaves the floors, lower its share and repeat. A turn keeps at least ⅓ (0.25 m radius).
6. **Pin the start** to the person's exact spot. The shift fades out over 5 m, or over 2.5 or 1.25 m if the longer fade would sweep the line across something solid.

`walkers.route_path` then resamples the line to 512 points. Each point's height comes from a downward ray from 0.6 m above its floor, averaged over ±0.5 m so walkers don't bob on the ruts.

### 6. The lanes: the fallback (`plan_loop`, `clear_run`, `loop_points`)

Without the scene's surfaces, or for a person who can't reach the main network, a straight lane: from where the game had the person (its pelvis over the lowest point of its feet), along its heading, a lane runs as far as:

- rays at 0.35, 1.0 and 1.7 m over the ground, at the centre and ±0.3 m to the sides, hit nothing (with 0.6 m to spare);
- downward rays every 0.25 m find ground, with no step over 30 cm;
- the same holds backwards;
- at most 15 m each way.

Other headings up to ±90° are tried, at a cost of 5 cm of lane per degree. Geometry within 15 cm of the start doesn't count: the game puts people where they brush things (`mesh_1519`'s head is under a ledge). A return lane 1 m to one side must be clear for the whole length; if neither side is, the person turns about on the spot. Under 2 m of lane in all, the person keeps its pose. The ray casts use the rip's surfaces, without the walkers. People lying on the ground count as obstacles. Snow drifts count as obstacles and as ground.

The loop (lane, half circle, lane, half circle) is resampled to 512 points with the ground height under each.

### 7. In Unreal (`M_GTB_Walker`)

| data | where | what |
|---|---|---|
| per vertex | UV1, UV2 | the 4 bone indices |
| | UV3, UV4 | the 4 weights (normalised) |
| | UV5, UV6.x | the bind position: Unreal cm, standing at the origin facing +X |
| per walker | `unreal/walkers/bones.png` → `T_GTB_WalkerBones` | one row per walker and bone (24 rows each), 3 texels per frame: the rows of the 3x4 matrix (R, t) from the bind position to the walking frame. RGB = R in −1..1, A = t in ±`BoneRange` (200 cm), both stored as (v + 1) / 2. The last (97th) frame holds the captured pose's rotations. Clips add a block of rows each, after the walk's (see Clips). |
| | `unreal/walkers/paths.png` → `T_GTB_WalkerPaths` | one row per walker: 512 + 1 points (x, y, z, yaw) along its loop, relative to the actor, in ±`PathRange` per channel; the last point is the first with yaw + 2π |
| per actor | custom primitive data 0–3 (`WalkerA`) | bone row, path row, phase at time 0, cycle time (× `WalkPeriod`; 0 = keep the pose) |
| | custom primitive data 4–7 (`WalkerB`) | stride (cm per cycle), loop length (cm), start (cm along the loop), 1 |

- **The textures** are 16-bit PNGs, imported as `TC_HDR_F32` with nearest filtering, no mips and no streaming. Half floats would round the paths to 2 cm. The shader reads them with `Texture.Load`.
- **The shader** (`gtb_hlsl.WALKER_WPO`): the cycle position is frac(phase + time / cycle). Each of the 4 bones' matrices is blended between two frames, the vertex is skinned in the walking frame, then placed and turned at its point along the loop: frac((start + progress × stride) / loop length). It returns that position minus the vertex's own, as World Position Offset.
- **Normals:** the mesh keeps the captured pose's normals and tangents, so the material turns them by R_now R_captured^T. The rotation's first and last columns go to the pixel shader through two VertexInterpolators (`QX`, `QZ`). `WALKER_ROT` turns the normal map's world normal, with `tangent_space_normal` off, and multiplies it by `TwoSidedSign`: the rip's people are wound inward (their normals, made from the winding, point into the body; 9–17% point out), and a world-space normal output skips the flip Unreal gives tangent-space normals on the back faces of two-sided materials. Without it every walker was lit inside out (rendered normals 155° from the true ones), dark when walking and nearly black once lying down to crawl; the static people never showed it. The snow reads the captured pose's vertex normal, so it stays on the clothes where the game had it (shoulders, caps) instead of settling on whatever faces up now, like a crawler's back.
- **Time** is `SceneTime + EngineTimeWeight × Time`, as in `Graph.time()`. The loop wraps, so engine time can grow without limit (unlike the smoke rise).
- **Bounds:** each walker mesh's bounds are extended over its whole loop (`walker_extent_cm`), so it isn't culled when it walks away from its spot.
- **Names:** each walker gets `MI_Walk_<its base material>` (e.g. `MI_Walk_M027`) with the base's values, so the people lying in the snow keep `MI_M027`. The `MI_M###` numbering skips the walker instances, so no other material is renumbered (`ue.live materials` points actors by name; see CLAUDE.md).

## Clips: animations from Mixamo

Any walker can play an animation clip instead of the walk, e.g. Mixamo's zombie crawl on `mesh_655` (a child), and swap back at any time.

### Add a clip

1. On [mixamo.com](https://www.mixamo.com) (an Adobe login), pick an animation. Leave **In Place** off, so the clip keeps its travel: that becomes the speed along the person's route. A clip made in place plays on the spot.
2. Download: **FBX Binary, Without Skin**, 30 fps, keyframe reduction none. Without Skin is the skeleton alone (Mixamo's own character; it doesn't matter which).
3. Save it as `animations/<name>.fbx` (e.g. `animations/zombie_crawl.fbx`). The folder is out of git: Mixamo's licence lets you use the animations in your work but not share the files. `gtb/fbx.py` reads them: binary FBX, no Blender needed. Checked against Blender 5.2's import of the zombie crawl: bone heads within 2 µm, and the retargeted person identical (0.00 mm).

Any FBX with Mixamo's bone names works (`mixamorig:Hips`, `LeftArm`, ...; the prefix doesn't matter).

Claude can do steps 1–3 in its browser pane: you log in to Mixamo there yourself (Claude never types passwords), name the animation you want, and Claude finds it, sets the download options and asks before each download.

### Which person

A person is named after its mesh, e.g. `mesh_655_655`. To find one, click the person in the Unreal viewport: the Outliner shows its name, in the `People` folder. `python -m ue.anim <capture>` lists everyone who can play a clip. Only people who walk can (on Frostpunk: 9 of 25).

### On another PC

Everything above works on a friend's PC from the project alone, with two differences:

- **Their own clips.** `animations/` isn't in git (Mixamo's licence doesn't allow sharing the files), so they download the clips they want from Mixamo with their own (free) Adobe account, into their own `animations/`.
- **Other names.** Their rip numbers its meshes in its own order, so their people have other names (your `mesh_655` is some other mesh there). They find theirs as above.

### Swap

With the editor open on the capture's level:

```bash
python -m ue.anim Frostpunk_20260929_115750 655 zombie_crawl
```

The person is `mesh_655_655`, `655`, or `all`; the clip is a file name in `animations/` or `walk`. `python -m ue.anim <capture>` lists the clips and who plays what.

- A clip already baked swaps instantly: only the actor's custom primitive data changes (one undo step, unsaved). The choice goes into `look.json` (`unreal.walkers.clips`, e.g. `{"mesh_655_655": "zombie_crawl"}`), so `gtb.py unreal` and `ue.live walkers` keep it.
- A clip new to `animations/` (or an editor whose walker data is older than the plan) first gets `python -m ue.live walkers <capture>` (about 2 minutes), which bakes every clip for every walker.
- In the editor, the walkers' clock is the editor's, hours by now. A person changing speed would jump along its route, so the swap moves its start along the route to keep it where it is. A Sequencer render, whose clock starts at 0, then starts that person elsewhere on its route; `ue.live walkers` puts the starts back.
- People who keep their pose (no room to walk, crutches, lying) play nothing.

The export prints `[people] mesh_655_655: ...; plays zombie_crawl` and `[people] clip zombie_crawl: 5.13 s loop, 1.96-3.06 m per cycle` (the stride scales with each person's hip height). `walker_check` prints `mesh_655_655: plays zombie_crawl, 1.96 m per 5.13 s cycle (0.38 m/s) | lowest point -0.0..+0.1 cm from the ground` and skips the walk's checks for it.

### How it works (`gtb/clips.py`)

- **Facing:** up is Blender's z, left runs from the right hip to the left; the clip is turned so its travel over a cycle points forward (x).
- **Loops:** a clip whose last frame repeats its first (within 5°) is a loop; the repeat is dropped and the travel measured over the whole cycle (Mixamo's zombie crawl: 154 frames at 30 fps, 2.15 m).
- **Bones:** each rig bone copies its Mixamo bone's turn away from Mixamo's rest pose, in world space: R = Q · A · H, where Q is Mixamo's world rotation now times its rest rotation's inverse, H turns the bind pose to the walking frame, and A matches the rests. Both rests stand upright facing forward, but Frostpunk binds in an A-pose with the forearms forward and Mixamo in a T-pose: so the upper arms, forearms, thighs and calves first turn (A) to point where Mixamo's do at rest, and the hands, fingers and feet take their parent's A. Without it, the arms are off by about 60°. Mapping: pelvis Hips, spine Spine, chest Spine2, neck Neck, head Head, clavicles Shoulder, upper arms Arm, forearms ForeArm, hands Hand, fingers HandMiddle1, thighs UpLeg, calves Leg, feet Foot.
- **Size:** the body's motion and the stride scale by hip height (the pelvis joint over the bind pose's lowest point, against Mixamo's hips over its lowest joint).
- **Ground:** the lowest point is put on the ground in every frame. The clip's own height, scaled, floated Frostpunk's bulkier child up to 12 cm in the flat part of the crawl.
- **Speed:** the hips' steady travel becomes the stride; what's left of their motion stays in the bones. The cycle time is the clip's own length (custom primitive data: seconds / `WalkPeriod`), and each person gets its own phase (`clip_phase`), so people playing one clip aren't in step.
- **Frames:** every clip is resampled to the walk's 96 frames (rotations slerped): 19 per second for a 5 s clip.

Not done: clips for the people lying down or on crutches (they have no route or bone UVs yet), foot or hand locks for clips (the planted hands of a crawl may slide a little where the proportions differ from Mixamo's), clips that leave the ground (a jump: the lowest point is held on the ground), blending between clips (a swap cuts).

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
- The walk:
  - **Each step surged** (the user: it's "like you use more force on each step for a moment ... then slow, then fast"). The first version kept the planted foot still by moving the whole body unevenly: its forward speed swung between 1.3 and 3.8 m/s in every step, and it swayed 14.6 cm. Now the body moves at a steady speed, and the legs are re-timed and locked instead.
  - **The stride was too long:** adding up, frame by frame, the faster of the two feet on the ground counted the double-support frames twice (2.35 m per cycle instead of about 2.2, with very uneven foot speeds). Each foot's own pace while down is the stride now.
  - **A little hop in every step:** with a steady body and the game's uneven foot speed, the locked foot got out of the leg's reach at the end of each step (by up to 6.8 cm), so both feet left the ground for a quarter of the cycle. Lowering the body to keep it in reach made it dip 15–22 cm. Re-timing the legs removed the cause, and a foot out of reach rolls onto its toes or heel before the body is lowered.
  - **The swinging foot scraped along the ground** at up to 8 m/s, which looks like sliding. The game's swing passes 1–5 cm over the ground, and a lock still fading out was counted as a foot that must reach the ground, which lowered the body. Only planted feet count now, and swinging feet are lifted 5 cm.
- **Walkers lit inside out** (fixed in master version 11): darker than the people around them, and nearly black once a clip laid one down. The people's normals come from their winding, which points inward, and the walker's world-space normal output skipped Unreal's two-sided flip. An earlier check blamed `mesh_655`'s darkness on the crater wall's shadow; that shadow is real, but the normals were the bigger part. Check with the normal buffer: a SceneCapture2D with `SCS_NORMAL` into an `RTF_RGBA32F` target, against the replayed pose's face normals turned to the camera (now a median of 23–25° apart, the rest is smooth normals and the normal map; 155° before).
- Two walkers stuck in place: one stood touching a ledge (fixed by ignoring geometry within 15 cm of the start), the other is boxed in.
- Path planning took minutes: rays tested against every triangle within 18 m. Each cast now uses a thin box around its line, so the export spends about 15 s on it.
- The editor capture camera inside a wall: place it by ray casts that see the loop.
- `unreal.KismetMaterialLibrary` doesn't exist in Python; it is `unreal.MaterialLibrary`.
- Routes:
  - **Road tiles failed as floors.** They have skirts reaching a metre into the terrain, and the rutted snow spans 40–70 cm of height inside one 0.5 m column. A floor rule that capped thickness rejected them; the step limit between neighbours does that job.
  - **The roads looked hatched.** The terrain pokes through the road surface along its triangle diagonals, so the top surface alone misses road columns. Roads are now the footprint of the road triangles.
  - **The city fell into hundreds of islands.** Forbidding floors next to anything solid cut every narrow gap. Tight floors are now usable at a high cost.
  - **Routes came back across open snow.** The way back paid 6 times as much on the way out's cells, so it left the road. Up and down one street, on two lanes, is what Frostpunk's spokes and rings suggest anyway.
  - **Walkers started 1–3 m off their spot.** The route began and ended at the person, so the start sat on a U-turn that smoothing pulled away. The person now starts mid-street, and the start is pinned.
  - **A boxed-in person walked out through a crate** (`mesh_637`), so the step to the network must cross floors.
  - **That check failed on its first metre:** the person's own column can be empty. Skip its own 0.3 m.
  - **Smoothing cut building corners** (2–6 points on 3 routes). The line is checked after smoothing and pulled back to the centre line.
  - **The far end behind wandered off the roads**, because only the way ahead preferred them.
  - **`walker_check` took 5 minutes:** one box around a whole 120 m route held most of the city's triangles. It now tests short stretches.
  - **Walkers zigzagged** (the user saw it in the viewport). The line had 30–50 cm kinks every metre or two: the grid's staircase left after a moving average, the lane offset switching between 0, ⅓, ⅔ and all of it from one point to the next, and corner fixes snapping points to the centre line. Straightened runs, rounded corners and a lane share smoothed along the way fixed it (strong turn flips: 3–19 per route before, 0–1 now).
  - **Walkers spun round on the spot** at a street end next to their start. The lane offset faded to 0 around the start, so the half-circle turn there had no radius, and the walker could turn either way. Pinning the start already puts the walker on its spot, so the fade went. The turns keep at least 0.25 m, and street ends make room for them.
  - **The loop's yaw closed with a guessed sign.** Unwrapping round the closed loop gives the exact total turn.

## Not done yet

- **Circuits:** everyone walks one street up and down. A loop round a block would need the way back to take another street at a fair price. The first try (6 times the cost on the way out's cells) sent people back across the open snow.
- **Junctions:** nobody walks through anyone now (two lanes), but walkers don't wait for each other where routes cross; `walker_check` lists pairs that come within 0.6 m.
- **`mesh_652`, `mesh_1519` and `mesh_637`** stand where the map has no way out. A finer grid near them, or letting a person step over a low crate, could free them.
- **Crutch walkers** (`mesh_642`, `mesh_649`) need their own cycle.
- **People at work:** every upright person on Frostpunk was walking. Someone standing still would walk too.
- **Blender** has no walking.

## Files

| path | what |
|---|---|
| `gtb/characters.py` | Skin data (`skin_arrays`, `backfill`), the rig, the bone solve (`Person`), the walk cycle (`fit_walk`, `Walk`) |
| `ue/walkers.py` | Finding the walkers, the baked cycle, their routes (or lanes), the textures and plan entries (with every clip's); `replay` mirrors the shader |
| `gtb/clips.py`, `gtb/fbx.py` | Animation clips: reading a binary FBX (no Blender), retargeting it onto a person |
| `ue/anim.py` | Swapping a person's clip in the open editor |
| `animations/` | The clips (`<name>.fbx`, out of git) |
| `ue/routes.py` | The walkable map (`WalkMap`), the roads (`road_meshes`), the routes (`plan_route`, `route_points`) |
| `ue/walker_check.py` | The checks above, and the routes map |
| `ue/editor/gtb_hlsl.py` | `WALKER_WPO`, `WALKER_ROT` |
| `ue/editor/gtb_materials.py` | `build_walker` (`M_GTB_Walker`: the surface master plus the walker parts) |
| `ue/editor/gtb_ue.py` | Imports the textures, sets `MI_Look_Walker` and each walker's custom primitive data and bounds |
| `ue/live.py` | `walkers`: the same, into an open editor |
| `ue/export.py` | Walker meshes get the bone UVs and `MI_Walk_*`; the solid surfaces with their materials go to the walkable map; the plan's `walkers` section |
| `profiles/frostpunk.json` | The `characters` rig |
| `tests/selftest.py` | A synthetic person (bones solved, planted foot, closed loop), a synthetic street (the route keeps to the road, round a building) and a synthetic Mixamo clip (T-pose rest retargeted onto the A-pose) |
