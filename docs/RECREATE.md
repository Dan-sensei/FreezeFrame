# Recreating the Frostpunk result from the Ninja Ripper rip

This runbook is for a fresh Claude session, or a person, on a machine that has this repo. It takes one Ninja Ripper frame rip to:

- a **Blender** scene (`captures/<name>/scene.blend`), and
- an **Unreal Engine** level (`unreal_project/`, `/Game/GTB/<name>/<name>`).

Both are matched to the game screenshot, with a night look and a day look. Read `CLAUDE.md` first: it holds the rules and the hard-won lessons, while this file holds the steps and the expected numbers.

The reference run was capture `Frostpunk_20260929_115750`, on 2026-09-29 (Windows 10, RTX 4090, 64 GB RAM). The whole runbook was then re-run on a fresh import of the same rip. The manifest came out identical and every step matched the numbers below.

## 0. What is not in git

| item | where it was | notes |
|---|---|---|
| The rip (5.6 GB) | `D:\NinjaRipperOutput\2026.09.29_11.51.52_Frostpunk.exe_69020\` | `frame_0000\` holds 6,195 `.nr` meshes, the `.dds` textures and `!screenshot.dds` (Ninja Ripper's own frame grab, which is the reference image). Copy the whole folder to the new machine. |
| Day reference | `captures/Frostpunk_20260929_115750/day_reference.webp` | An official Frostpunk day screenshot (1280×720) supplied by the user. It is a different shot, used only to judge the day look by eye. |
| `captures/`, `config.json`, `unreal_project/` | generated | They are rebuilt by the steps below. |

Without the rip, nothing about the scene can be recreated. The code, the profile (`profiles/frostpunk.json`) and the looks (`profiles/looks/`) are all in git.

## 1. Tools

| tool | tested version | notes |
|---|---|---|
| Python | 3.13.1 (3.11+ works) | `python -m pip install -r requirements.txt` (numpy, pillow ≥ 11, opencv-python, pynput, opencolorio) |
| Blender | 5.2 (optional for Unreal) | For the Blender scene (step 3). `blender_exe` in `config.json` (default `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`). Always run it with `--factory-startup`. **Unreal only:** skip Blender, and in step 3 run `python gtb.py process Frostpunk_20260929_115750` instead of `all`. The colour LUT then uses Blender 5.2's colour config, downloaded once (about 4 MB) into `.cache/`, and gives the same colours. After the Unreal build, `python gtb.py unreal-calibrate Frostpunk_20260929_115750` takes the place of Blender's exposure calibration (not needed with `frostpunk_day`, which has its own exposure). |
| Unreal Engine | 5.8.3 (Epic launcher install) | Found automatically under `C:\Program Files\Epic Games\UE_5.x`, or set `unreal_editor` in `config.json` to `...\Engine\Binaries\Win64\UnrealEditor-Cmd.exe`. The project that `gtb.py` generates turns on the Python, Editor Scripting, Movie Render Queue, Sequencer Scripting, Niagara Fluids, Toolset Registry and Niagara Toolsets plugins. |
| Ninja Ripper | 2.18 | Only needed for new captures. Settings are in `CLAUDE.md` (First run on a new PC). |
| GPU | DX12, SM6, ray tracing | The Unreal project uses Lumen with hardware ray tracing. |

Check the install before touching the capture. Every line of the self-test must say `PASS` (30 checks; without Blender, the two that need it say `SKIP`):

```bash
python tests/selftest.py
```

## 2. Make the capture folder

**From the existing rip** (this recreates the reference run):

```bash
python gtb.py import "D:\NinjaRipperOutput\2026.09.29_11.51.52_Frostpunk.exe_69020" --name Frostpunk_20260929_115750
```

The command takes the newest `frame_*` folder and reads the exe and resolution (`Frostpunk.exe 3440x1440`). It writes `captures/<name>/capture.json`, which points at the rip. Copy `day_reference.webp` into the capture folder if you have it.

**A new rip instead**: run `python gtb.py watch`, launch the game through Ninja Ripper (for Steam, fully exit Steam first), hide the HUD and press PrintScreen. The daemon creates the capture folder and runs step 3 by itself.

## 3. Blender: process, build, calibrate (about 8 min)

Without Blender (Unreal only), run `python gtb.py process Frostpunk_20260929_115750` instead: the same console lines up to the texture ones, then go to the Unreal step. Exposure: Blender's `calibrate` matches the look's exposure to the game screenshot. Without Blender, run `python gtb.py unreal-calibrate <capture>` after `unreal` (the editor closed; up to 4 headless renders of about 40 s): it moves `look.unreal.exposure_offset`, which Unreal applies before the colour transform, like Blender's exposure. Looks with their own exposure don't need it: `frostpunk_day` (−1.4, `calibrate: false`) is one; `frostpunk_night` and a new capture's default look are not. `gtb.py unreal` reminds you.

```bash
python gtb.py all Frostpunk_20260929_115750
```

This runs `process` (rip → `manifest.json` + meshes + textures), `build` (→ `scene.blend`) and `calibrate` (matches exposure to the screenshot). Expected console lines (the reference values):

- `solved FOV from 690 rigid draws: fov_y=55.00 deg, aspect=2.389`
- `converting 173 textures`. One of them is auto-detected as a shared engine texture and ignored.
- `170 of 173 textures match the known-texture database (frostpunk_textures.json): they get its checked roles.` This rip is the database's reference, so all its hashable textures match. On another Frostpunk rip the count is lower: the rest are classified by the rules, and `python gtb.py audit <capture>` lists them for checking (see `CLAUDE.md`, "Known textures and the audit"). The database already holds the corrections the rules make on this rip (`t0091` an albedo, `t0018`, `t0084` and `t0159` normals, the `t0033` atlas a detail layer), so their `texture t####: ... -> ...` lines no longer print here.
- `per-mesh winding: flipped 1353 meshes to match the game's normals`
- `1 smoke plume(s) from smoke columns: ['mesh_5092_5092']` (the generator)
- `7 decal volume(s) hidden (boxes that project a texture)`: the snow-stamp boxes on the cliff tops (`mesh_1591`–`1598`, `1601`).
- `2 overlay layer(s) hidden (only a mask texture, blended over the ground)`: the frost-streak ring around the crater (`mesh_5267`, `mesh_5269`).
- `137 surface draws far below the scene hidden (another render pass)`: blotchy strips and a snow ring 0.7–2 km under the city.
- `2 game light(s) far below the scene switched off` (`GL0000`, `GL0001`, 750–850 m away).
- `wrote manifest with 2302 meshes {'surface': 1539, 'effect': 719, 'sprite': 44}`
- The manifest has 384 lights (323 point and 61 hemi).
- `built 1539 + 719 hidden effect objects, 44 particle sprites, 111 materials`
- The new capture starts with the profile's look, which is the night look (`profiles/looks/frostpunk_night.json` without an exposure). `calibrate` should end at `exposure` −0.73 ± 0.05: the verification run got −0.73, and the original capture −0.76 under older material rules.

Keep that calibrated night look under its own name, because step 5 replaces `look.json`:

```bash
copy captures\Frostpunk_20260929_115750\look.json captures\Frostpunk_20260929_115750\look_night.json
```

Then render the close-up checks:

```bash
python gtb.py closeups Frostpunk_20260929_115750
```

## 4. Unreal: build, render, compare (about 2.5 min)

Close the Unreal editor first; `gtb.py` refuses to write while it has the project open.

```bash
python gtb.py unreal Frostpunk_20260929_115750
```

The first ever run also compiles shaders and textures, which takes a few minutes more. Expected output:

- `plan: {'surface': 1298, 'cloth': 44, 'walker': 11, 'effect': 716, 'sprite': 47, 'snowdrift': 241}, 117 materials, ... 384 game lights`. That is one more material than Blender's 111 because the ground-mist cards get their own (`MI_Sprite_mist_t0114`), plus 5 `MI_Walk_*` for the walking people. The other mist materials replace ones Blender also has. The 3 extra sprites are mist layers that Blender keeps hidden (see step 6). `cloth` counts the red banners that flutter, and `walker` the people who walk (both are also counted in `surface`). The export ray-casts each banner's clearance to the walls and frames around it, in a few seconds. `python -m ue.cloth_check Frostpunk_20260929_115750` must then print `0 banner(s) clip, 0 jerky, 4 barely move`.
- People, before the plan line (about 50 s): `[people] 25 people: 11 walk, 2 upright on a crutch and 12 lying keep their pose`, `walk cycle from 11 walkers: legs within 3.3 deg rms`, `roads: MI_M000 (267 meshes; 5 of 11 people stand on it)`, `walkable map: 205191 floors (122705 in the main network, ...)`, one line per walker (8 streets of 60–115 m, 82–100% on roads, and `mesh_652_652: 2.6 m lane`; stride 2.1–2.3 m for adults and 1.5 m for children), and `2 with no room to walk keep their pose: ['mesh_1519_1519', 'mesh_637_637']`. The first export of a capture processed before 2026-10-01 reads the people's bind pose from the rip once (`skin data of 263 skinned meshes read from the rip`); without the rip, nobody walks.
- The build takes about 60 s, with `smoke columns: 1` and `done (0 warnings)`.
- `4 frame(s)` rendered by Movie Render Queue in about 25 s.
- Night look: about 0.1 stops from the screenshot and from Blender's game view. Edge alignment against Blender is only 0.6–0.75, because the dense night snowfall uses different random flakes in each engine.
- Day look (step 5): within about 0.05 stops of Blender's game view, with edge alignment about 0.93.

Everything lands in `captures/<name>/unreal/`:
- `parity.png`: game | Blender | Unreal, for the game camera and three close-ups. This is the main check.
- `comparison.png`: Unreal against the screenshot.
- `metrics.json`, and the stills in `renders/stills/`.

To look around in the editor:

```bash
python gtb.py unreal-open Frostpunk_20260929_115750
```

To check the 250-frame animation, run `python gtb.py unreal-render Frostpunk_20260929_115750 --anim`; frames go to `unreal/renders/anim/`.

### 4b. The generator's smoke: ask the user first (editor open, about 1 min)

**Ask the user which generator smoke they want before going on; don't choose for them.** The build ends by printing the choice (`GENERATOR SMOKE - ask the user ...`):
- **(a)** the sprite column the build just made: darker puffs, lit by the fire, flames at the base.
- **(b)** a Niagara Fluids plume: light, billowing smoke rising out of the furnace, the look the original user tuned.

They can switch any time. For (a) there's nothing to do. For (b), with the editor open on the level (`gtb.py unreal-open`), run:

```bash
python -m ue.fluid_smoke Frostpunk_20260929_115750
```

It prints `GTB fluid smoke: GeneratorPlume at (-5031, -8483, -12311), ...`, hides the sprite column `SmokeColumn00`, and fills in about 30 s. Then File → Save All. Details, settings and pitfalls: [GENERATOR_SMOKE.md](GENERATOR_SMOKE.md).

Switching between the two later changes visibility only:

```bash
python -m ue.fluid_smoke Frostpunk_20260929_115750 --show sprites
```

`--show fluid` switches back.

## 5. The day look

```bash
python gtb.py look Frostpunk_20260929_115750 frostpunk_day
```

```bash
python gtb.py unreal-look Frostpunk_20260929_115750
```

The first command copies the preset into `look.json` and renders Blender. It saves `scene.blend`, so close it in Blender first. The day look has `calibrate: false`, because it was tuned by eye against `day_reference.webp`. The second command pushes the same look to Unreal without re-importing, then renders and compares.

To go back to night, copy `look_night.json` over `look.json` and run `gtb.py render <name> --save` and `gtb.py unreal-look <name>`. The reference machine was left on the day look.

## 6. What "right" looks like (judge the images, not only the numbers)

- **Framing:** the game camera matches the screenshot edge for edge (edge alignment against Blender is about 0.93–0.95).
- **Snow:** roof snow sits on up-facing surfaces in patches. Around the crater rim you see the snowy cliff tops and the mining frame on the wall (`mesh_660`). A flat white sheet over them means the frost-streak overlay ring (`mesh_5267`/`5269`) is being drawn solid; it should be hidden. No large snow cubes on the cliff tops: those were decal boxes (`mesh_1591`–`1598`, `1601`), now hidden. Black-and-white leopard spots there mean a grayscale mask was used as colour; see `assign_slots` in `gtb/scene_common.py`.
- **Terrain:** the crater floor has no UVs and uses the flat `materials.ground_color`.
- **Dead trees** (`MI_M023`, 61 meshes): grey bark (`t0046`) under frost, lying in the snow. No yellow-green or marble patches.
- **Wood and metal:** the frosted planks of the banner stands, towers and building frames (`MI_M008`: colour `t0017`, normal `t0018`) are weathered grey-brown wood with frost. Green, orange or blue streaks mean a packed normal map is being used as colour; see step 8.
- **Lights:** street lamps and building lights glow warm (384 game lights). The generator glows.
- **Smoke:** chimney smoke sprites sit at the chimneys. Smoke puffs rise and fade in a loop in the Unreal editor. In Unreal, after step 4b, the generator has a billowing bluish-white fluid plume (`GeneratorPlume`). It rises out of the furnace with an orange glow at its base, and thins and lightens as it climbs. It rises straight, with no sideways drift and no hard edge. Without step 4b, or after `--show sprites`, it has the sprite column instead (`SmokeColumn00`): darker and denser, lit by the fire, with flames at its base. Blender's volumetric version (`smoke.enabled`) is off in both presets.
- **Mist and haze (Unreal):** the game's mist cards drift in a slow loop, all through `MI_Look_Mist`. They fade out near buildings, the ground and the camera, so no card cuts a hard line through anything. For more or less mist, change `look.unreal.mist.opacity`.
  - Ground mist over the city: 8 meshes of soft 75–80 m cards (`mesh_5227`, `5235`, `5236`, `5239`, `5244`, `5270`–`5272`, `MI_Sprite_mist_t0114`).
  - Haze on the crater walls: `mesh_5266` (18–23 m cards, `MI_Sprite_mist_t0128`). It softens the blue ice walls, most visibly the lower-right wall seen from the game camera.
  - Wind-blown haze over the ice east of the city (`mesh_5273`, `MI_Sprite_mist_t0105`) and wisps over the east crater wall (`mesh_5255`, `mesh_5268`, `MI_Sprite_mist_t0126`), both subtle. `process` drops these three as snowflakes, and the Unreal export brings them back as mist.
  - Still hidden on purpose: the game's snowflake particles (the procedural snowfall replaces them), plus lamp glows, small flames and building steam, which have no Unreal look yet. The list is in `CLAUDE.md`, "What in the rip isn't the scene".
- **Snowfall:** flakes fall around the camera in stills and in the animation.
- **Banners (Unreal only):** the 44 red banners (`MI_M002` → `MI_Look_Cloth`) flutter in the snow's wind. Their top part stays still.
  - Free-hanging banners (like `mesh_1004`) swing downwind with a ripple running down them.
  - The 30 whose bottom runs through their stand's bottom block only ripple, fading out before the bottom, with no bow in the middle.
  - None passes through a wall, beam or pole, and none snaps.
  - A zigzag along a long banner means `look.unreal.cloth.wavelength` is too short (keep it at 0.6 or more).
  - With the editor open, `python -m ue.live cloth Frostpunk_20260929_115750` re-applies them.
- **People (Unreal only; all details in [WALKING_PEOPLE.md](WALKING_PEOPLE.md)):** 8 of the people walk up and down the city's streets on the game's roads, 60–115 m each way, at a steady pace, keeping to the right so people coming the other way pass. A ninth (`mesh_652`) walks a short lane on the spot it has. They walk in the stride the game's own walkers have: heel strike, the swinging knee bent, arms swinging opposite the legs, the planted foot still (it slips less than 1 cm/s). At time 0 each stands within about half a metre of where the game had it; in the editor viewport they walk all the time. Children step faster. The 12 people lying in the snow, the 2 on crutches (`mesh_642`, `mesh_649`) and the 2 in cramped spots (`mesh_1519` under a ledge, `mesh_637` between a crate and a wall) keep their pose.
  - Each walker's mesh has its own `MI_Walk_*` instance (the base material's values) on `MI_Look_Walker`, and sits in the `People` folder.
  - `MI_Look_Walker`'s `Walk` = 0 stops them all (they stand in the game's pose); `WalkPeriod` is an adult's seconds per cycle (two steps). Both come from `look.unreal.walkers`.
  - With the editor open, `python -m ue.live walkers Frostpunk_20260929_115750` adds or refreshes them (about 2 minutes) and touches nothing else.
  - **Animations from Mixamo (optional):** any walking person can play a Mixamo animation instead of the walk, e.g. a zombie crawl. Download it as FBX Binary, Without Skin, into `animations/<name>.fbx` (not in git: get your own copy with a free Adobe account), click the person in the viewport to read its name in the Outliner, then `python -m ue.anim Frostpunk_20260929_115750 <person> <name>` (`walk` puts the walk back). Everything is in [WALKING_PEOPLE.md](WALKING_PEOPLE.md), "Clips".
  - `python -m ue.walker_check Frostpunk_20260929_115750` must end with `9 walk, 2 keep their pose, 2 pair(s) walk through each other` (`mesh_646` crossing two others at a junction for a moment), with the body speed at 100–100% of the average (no surge in each step) and planted feet slipping about 1 cm/s. `captures/<name>/unreal/walker_check/routes.png` shows the routes on the streets: smooth two-lane lines with half-circle turns at the ends, no zigzags.
- **Close-ups:** these are dollied from the game camera. Unreal's close-ups are a little darker than Blender's (up to about 0.4 stops) because Lumen occludes sky light, which EEVEE doesn't. That is expected.

## 7. Where things are

| path | what |
|---|---|
| `gtb/process.py`, `gtb/nr.py`, `gtb/textures.py` | Rip → manifest: camera solve, draw classification, textures, lights, winding, sprites, snow box, smoke plumes |
| `gtb/audit.py` | `gtb.py audit`: material sheets (textures, roles, game screenshot crops) and a report of what to check |
| `profiles/frostpunk_textures.json` | Known textures: mip hashes and checked roles of the reference capture's 170 textures ([KNOWN_TEXTURES.md](KNOWN_TEXTURES.md)) |
| `gtb/scene_common.py` | Rules shared by both engines: default look, which texture is albedo, normal or mask |
| `blender/` | Blender builder (`gtb_scene.py`) and the build, render and close-up scripts |
| `ue/export.py`, `ue/look.py` | Manifest → Unreal plan and glb files; `look.json` → Unreal values |
| `ue/colour_lut.py` | Bakes Blender's grade and AgX into the LUT that Unreal applies, without Blender (OpenColorIO with Blender's colour config; `--compare-blender` checks it against `ue/blender_lut.py`, Blender's own bake) |
| `ue/editor/` | Runs inside Unreal: master materials (`gtb_hlsl.py`, `gtb_materials.py`) and the level build (`gtb_ue.py`) |
| `ue/pipeline.py` | Runs Unreal headless (commandlet, then Movie Render Queue) and writes the comparison sheets |
| `gtb/characters.py` | People: each one's bones solved from its skinned draw, the walk cycle fitted to the walkers |
| `ue/walkers.py`, `ue/routes.py`, `ue/walker_check.py` | Walking people for Unreal: the baked cycle, routes along the city's streets (a walkable map from the rip), the two data textures, a numpy replay of `M_GTB_Walker`, and its checks ([WALKING_PEOPLE.md](WALKING_PEOPLE.md)) |
| `gtb/clips.py`, `gtb/fbx.py`, `ue/anim.py` | Animation clips (Mixamo FBX in `animations/`, not in git): read without Blender, retargeted onto each walker, swapped live with `python -m ue.anim` |
| `ue/cloth_check.py` | Replays the banner flutter in numpy and reports banners that move into geometry |
| `ue/live.py` | Pushes the banners, all material instances, the sprite column, the sprites or the walking people into the open editor (`cloth` / `materials` / `plume` / `sprites` / `walkers`) |
| `ue/fluid_smoke.py` | Builds the generator's Niagara Fluids smoke in the open editor ([GENERATOR_SMOKE.md](GENERATOR_SMOKE.md)) |
| `ue/remote.py` | Runs a Python file inside an open Unreal editor, for live tweaks (needs Python Remote Execution turned on) |
| `profiles/frostpunk.json`, `profiles/looks/` | Game profile, night and day looks |

## 8. If something differs

- **Different mesh or texture counts after `process`:** check that the rip folder is complete (6,195 `.nr` files) and that `profiles/frostpunk.json` is unchanged.
- **FOV not solved:** pre-VS data is missing. Ninja Ripper needs "save pre-VS" on.
- **Textures or colours look wrong on your own rip** (another frame, other buildings): don't guess from the viewport. Run `python gtb.py audit <capture>` and read `captures/<name>/audit/report.md` and the `materials_NN.png` sheets. Textures marked **K** matched the known-texture database and have checked roles; the ones marked **?** were guessed and are listed first, biggest on screen first, each beside a crop of the game screenshot. `python gtb.py known <capture>` lists the textures the database doesn't have, and `python gtb.py known <capture> <texture id> --role <role>` records one you've checked. Fix what's wrong as `CLAUDE.md` describes ("Known textures and the audit"), then `python gtb.py textures <capture>` and `python -m ue.live materials <capture>`. Material numbers differ between rips; the report says which reference material has the same albedo.
- **Green, orange or blue streaked surfaces** (or flat, texture-less ones): a normal map is being used as colour, or the other way round. **Yellow-green or marbled patches on trees and rocks**: the `t0033` atlas is being used as colour. Captures processed before these texture-role fixes (2026-09-29) need `python gtb.py textures <capture>` once. It needs no rip and should change 5 roles on this capture: `t0018`, `t0084` and `t0159` become normals, `t0091` an albedo, and `t0033` a detail layer. Running it again changes nothing. Then run `build` (close `scene.blend` first) and `unreal`, or `python -m ue.live materials <capture>` with the editor open.
- **A grey gap in the terrain or a cliff, seen from a free camera:** the game skipped terrain chunks outside its view, so they aren't in the rip (Blender has the same gaps). Nothing was deleted: compare the level with `plan.json`. See "What in the rip isn't the scene" in `CLAUDE.md`.
- **Lights hanging over empty snow** (e.g. `GL0011`): the game's own fill lights, with no visible fixture. The user chose to keep them.
- **Big snow-coloured cubes on the cliff tops:** deferred decal volumes drawn as solid boxes. `scene_common.decal_volume` hides them in `process` and in the export, and `python -m ue.live materials <capture>` hides them in an open editor.
- **Blotchy black-and-white strips or a snow ring floating far away:** geometry from another render pass, placed 0.7–2 km under the city. `scene_common.below_scene` hides it. For a capture processed before that rule, the Unreal export hides it anyway, and `python -m ue.live materials <capture>` hides it in an open editor.
- **No haze on the crater walls or over the ice, or `mesh_5273`, `mesh_5255` and `mesh_5268` sit in "Effects (hidden)" in the editor:** the level was built before the mist rules (2026-09-30). With the editor open, run `python -m ue.live sprites <capture>`: it re-imports those three meshes as sprites and moves all four to the mist look. The Content Browser then jumps to `SM_01646_mesh_5273_5273`; that's expected. Undo reverts it, and File → Save All keeps it. A fresh `gtb.py unreal` builds them directly.
- **Unreal step fails:** read `captures/<name>/unreal/build.log`, then `build_engine.log`. For renders, read `render_stills_engine.log`.
- **Unreal colours about a stop too bright or washed out:** the colour pass must stay before the tonemapper. See the Unreal section of `CLAUDE.md`.
- **Nobody walks:** the export prints `[people]` lines; without them the profile has no `characters` rig, or the capture has no skin data and its rip is gone (`no skin data and the rip is gone`). A level built before 2026-10-01 has the people frozen: with the editor open, run `python -m ue.live walkers <capture>`. People that vanish mean `M_GTB_Walker` didn't compile (the live command prints its instruction counts). Feet that slide or walkers in walls: `python -m ue.walker_check <capture>`, and [WALKING_PEOPLE.md](WALKING_PEOPLE.md), "Check it". People that stand still while others walk are the crutch users and those with no room (see step 6), or `MI_Look_Walker`'s `Walk` is 0.
- **No generator smoke after step 4b:** give it 60–90 s if the editor isn't in focus. If `GeneratorPlume` has no Niagara system (its Details show an empty Niagara System asset), its assets were never saved. `python -m ue.fluid_smoke <capture> --show fluid` rebuilds it, and so does the build command. Then check that the sky dome has no collision (the script sets its profile to `NoCollision`) and that the three plugins are enabled. See [GENERATOR_SMOKE.md](GENERATOR_SMOKE.md), Pitfalls.
- **Anything else:** use the lessons list in `CLAUDE.md`. Fix things in code or the profile and re-run the step, never by hand in the `.blend` or the level.
