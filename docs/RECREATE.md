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
| Python | 3.13.1 (3.11+ works) | `python -m pip install -r requirements.txt` (numpy, pillow ≥ 11, opencv-python, pynput) |
| Blender | 5.2 | `blender_exe` in `config.json` (default `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`). Always run it with `--factory-startup`. |
| Unreal Engine | 5.8.3 (Epic launcher install) | Found automatically under `C:\Program Files\Epic Games\UE_5.x`, or set `unreal_editor` in `config.json` to `...\Engine\Binaries\Win64\UnrealEditor-Cmd.exe`. The project that `gtb.py` generates turns on the Python, Editor Scripting, Movie Render Queue and Sequencer Scripting plugins. |
| Ninja Ripper | 2.18 | Only needed for new captures. Settings are in `CLAUDE.md` (First run on a new PC). |
| GPU | DX12, SM6, ray tracing | The Unreal project uses Lumen with hardware ray tracing. |

Check the install before touching the capture. The self-test must print `PASS` 12 times:

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

```bash
python gtb.py all Frostpunk_20260929_115750
```

This runs `process` (rip → `manifest.json` + meshes + textures), `build` (→ `scene.blend`) and `calibrate` (matches exposure to the screenshot). Expected console lines (the reference values):

- `solved FOV from 1041 rigid draws: fov_y=55.00 deg, aspect=2.389`
- `220 textures`. One of them is auto-detected as a shared engine texture and ignored.
- `texture t0091: normal:AG -> albedo` and `texture t0159: albedo -> normal:AG`: the shader slot consensus correcting two roles. `t0018` and `t0084` are already classified as normals. See the texture-role note in `CLAUDE.md`.
- `per-mesh winding: flipped 1353 meshes to match the game's normals`
- `1 smoke plume(s) from smoke columns: ['mesh_5092_5092']` (the generator)
- `wrote manifest with 2302 meshes {'surface': 1685, 'effect': 573, 'sprite': 44}`
- The manifest has 384 lights (323 point and 61 hemi).
- `built 1685 + 573 hidden effect objects, 44 particle sprites, 112 materials`
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

- `plan: {'surface': 1444, 'cloth': 44, 'effect': 573, 'sprite': 44, 'snowdrift': 241}, 112 materials, ... 384 game lights`. `cloth` counts the red banners that flutter (they are also counted in `surface`). The export ray-casts each banner's clearance to the walls and frames around it, in a few seconds. `python -m ue.cloth_check Frostpunk_20260929_115750` must then print `0 banner(s) clip, 0 jerky, 4 barely move`.
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
- **Snow:** roof snow sits on up-facing surfaces in patches. The rim plateau (mesh_5267/5269) and the far cliff tops (mesh_1592–1598) are **plain snow**. Black-and-white leopard spots there mean a grayscale mask was used as colour; see `assign_slots` in `gtb/scene_common.py`.
- **Terrain:** the crater floor has no UVs and uses the flat `materials.ground_color`.
- **Wood and metal:** the frosted planks of the banner stands, towers and building frames (`MI_M008`: colour `t0017`, normal `t0018`) are weathered grey-brown wood with frost. Green, orange or blue streaks mean a packed normal map is being used as colour; see step 8.
- **Lights:** street lamps and building lights glow warm (384 game lights). The generator glows.
- **Smoke:** chimney smoke sprites sit at the chimneys. The generator has a tall dark smoke column (Unreal only, `SmokeColumn00`). Blender's volumetric version (`smoke.enabled`) is off in both presets. Smoke puffs rise and fade in a loop in the Unreal editor.
- **Snowfall:** flakes fall around the camera in stills and in the animation.
- **Banners (Unreal only):** the 44 red banners (`MI_M002` → `MI_Look_Cloth`) flutter in the snow's wind. Their top part stays still.
  - Free-hanging banners (like `mesh_1004`) swing downwind with a ripple running down them.
  - The 30 whose bottom runs through their stand's bottom block only ripple, fading out before the bottom, with no bow in the middle.
  - None passes through a wall, beam or pole, and none snaps.
  - A zigzag along a long banner means `look.unreal.cloth.wavelength` is too short (keep it at 0.6 or more).
  - With the editor open, `python -m ue.live cloth Frostpunk_20260929_115750` re-applies them.
- **Close-ups:** these are dollied from the game camera. Unreal's close-ups are a little darker than Blender's (up to about 0.4 stops) because Lumen occludes sky light, which EEVEE doesn't. That is expected.

## 7. Where things are

| path | what |
|---|---|
| `gtb/process.py`, `gtb/nr.py`, `gtb/textures.py` | Rip → manifest: camera solve, draw classification, textures, lights, winding, sprites, snow box, smoke plumes |
| `gtb/scene_common.py` | Rules shared by both engines: default look, which texture is albedo, normal or mask |
| `blender/` | Blender builder (`gtb_scene.py`) and the build, render and close-up scripts |
| `ue/export.py`, `ue/look.py` | Manifest → Unreal plan and glb files; `look.json` → Unreal values |
| `ue/blender_lut.py` | Blender bakes its own grade and AgX into the LUT that Unreal applies |
| `ue/editor/` | Runs inside Unreal: master materials (`gtb_hlsl.py`, `gtb_materials.py`) and the level build (`gtb_ue.py`) |
| `ue/pipeline.py` | Runs Unreal headless (commandlet, then Movie Render Queue) and writes the comparison sheets |
| `ue/cloth_check.py` | Replays the banner flutter in numpy and reports banners that move into geometry |
| `ue/live.py` | Pushes the banners or all material instances into the open editor (`cloth` / `materials`) |
| `ue/remote.py` | Runs a Python file inside an open Unreal editor, for live tweaks (needs Python Remote Execution turned on) |
| `profiles/frostpunk.json`, `profiles/looks/` | Game profile, night and day looks |

## 8. If something differs

- **Different mesh or texture counts after `process`:** check that the rip folder is complete (6,195 `.nr` files) and that `profiles/frostpunk.json` is unchanged.
- **FOV not solved:** pre-VS data is missing. Ninja Ripper needs "save pre-VS" on.
- **Green, orange or blue streaked surfaces** (or flat, texture-less ones): a normal map is being used as colour, or the other way round. Captures processed before the texture-role fix (2026-09-29) need `python gtb.py textures <capture>` once. It needs no rip and should change 4 roles on this capture. Then run `build` (close `scene.blend` first) and `unreal`, or `python -m ue.live materials <capture>` with the editor open.
- **Unreal step fails:** read `captures/<name>/unreal/build.log`, then `build_engine.log`. For renders, read `render_stills_engine.log`.
- **Unreal colours about a stop too bright or washed out:** the colour pass must stay before the tonemapper. See the Unreal section of `CLAUDE.md`.
- **Anything else:** use the lessons list in `CLAUDE.md`. Fix things in code or the profile and re-run the step, never by hand in the `.blend` or the level.
