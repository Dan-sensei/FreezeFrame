# GameToBlender: notes for Claude

Pipeline: Ninja Ripper 2 frame rip of a DX11 game → `gtb/process.py` (parse .nr, solve camera, classify draws/textures) → `blender/gtb_scene.py` (Blender 5.2 scene) → tune `look.json` against the game screenshot. See README.md for user-facing usage.

Tested end to end on **Frostpunk 1** with Ninja Ripper 2.18 and Blender 5.2 (2026-09-29). Sekiro has a starting profile but no real capture yet.

## First run on a new PC
- `captures/` and `config.json` are not in git. `config.json` is created on first run: check `blender_exe` (default `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`) and that `ripper_output_dir` ("auto") resolves to Ninja Ripper's output folder (`python gtb.py watch` prints it).
- `python -m pip install -r requirements.txt`, then `python tests/selftest.py` must print all PASS before a real capture.
- Ninja Ripper settings that worked: frame rip = PrintScreen, textures DDS, save pre-VS and post-VS meshes, screenshots on (`%PUBLIC%\ninjaripper\nr218\nrconfig.xml`). The output folder needs space (~5 GB per Frostpunk frame).
- The first Frostpunk capture should need little manual work thanks to `profiles/frostpunk.json`. If something looks wrong, diagnose from the data (per-shader groups, texture contact sheets, ray casts in Blender) and put the fix in code or the profile, not in the .blend.

## Working with the user
- Judge results against the user's own screenshots and game references, not only the metrics. Users care about close-up shots, not just the game camera view, so always check `python gtb.py closeups <capture>` renders too.
- Use stills **and** animation: snow/smoke are driven by the frame number (no baking).
- Never write to a `scene.blend` the user has open. Ask first, or build to another name: `blender -b --factory-startup --python blender/build.py -- <capture> other.blend`.
- Always run Blender with `--factory-startup`, because user add-ons can spawn processes and flood logs.
- Blender 5.x API changed: compositor = `scene.compositing_node_group`; GN modifier inputs = `mod.properties.inputs.<Socket_N>.value`; sky types `SINGLE_SCATTERING`/`MULTIPLE_SCATTERING`; GN uses `FunctionNodeSeparateColor`. Probe with a tiny script before assuming.

## Capturing (what went wrong the first time)
- Steam games: **fully exit Steam** (tray → Exit), run `steam.exe` from Ninja Ripper, then start the game. If Steam was already running, the hook never reaches the game. Check: the game process must have `Ninja Ripper\\...\\intruder.dll` loaded.
- Ninja Ripper makes one folder per hooked process (`<date>_<exe>_<pid>`) at process start; each rip lands in `frame_NNNN` inside it. The daemon watches for new `.nr` files, not new folders.
- Rip = PrintScreen. Don't close the game until the daemon says `rip complete` (a Frostpunk frame is ~6,000 files / 5 GB).
- Borderless windowed, overlays off, HUD hidden if possible. NR also saves its own `!screenshot.dds`, which is used as the reference.

## Ninja Ripper 2.18 data (differs from older docs)
- Each `.nr` holds 2 GEOMs (pre_vs + vs) with the same data, so we use the vs one.
- Post-VS positions are **expanded** (one vertex per index); paired to pre-VS rows via the index buffer. This makes the FOV solve exact (Frostpunk: 55.00° vertical).
- No render-state props. Shadow maps = orthographic (w ≡ 1). Deferred light volumes/decals sample **screen-sized** textures and become the 384 recovered game lights (480-tri spheres). `TEXTURE_SAVING_FAILED` refs = live render targets (e.g. the generator smoke column).
- **Winding varies per shader.** Decide per mesh from the game's vertex normals (1,353 of 1,685 Frostpunk meshes needed flipping). A single global rule looked fine (custom normals hid it) but broke roof snow, backfaces and the cliffs.

## Frostpunk specifics (encoded in profiles/frostpunk.json)
- In-world UI (building icons): pre-VS layout `COLOR0,POSITION0,TEXCOORD0` → skip.
- Particles: layout `POSITION0,COLOR0,TEXCOORD0,TEXCOORD1,COLOR1,NORMAL0` = smoke/steam flipbooks (RGB = normal map, A = shape) + fire flipbooks + snow specks. Rebuilt as camera-facing sprites; specks dropped (the procedural snowfall replaces them).
- Materials: albedo alpha is a **mask, not opacity**; normals packed in A+G (R ≈ roughness); one engine-wide snow texture is bound to most shaders (auto-detected as "shared" and ignored). Meshes whose *only* texture is that one are snow drifts, which get a snow material. Terrain has no UVs and gets a flat colour. Shader families order slots differently, so there are no fixed slot numbers.
- The game adds roof snow in-shader, so we use `add_surface_snow` (`materials.snow_cover` / `snow_threshold`).
- Looks: `profiles/looks/frostpunk_night.json` (tuned to the capture screenshot, auto-exposure) and `frostpunk_day.json` (tuned by eye to an official day screenshot; `calibrate: false`). Apply with `python gtb.py look latest frostpunk_day`.

## Tuning loop
Edit `captures/<name>/look.json`, then `python gtb.py render <capture> --save` (writes comparison.png + metrics) and `python gtb.py closeups <capture>`. `calibrate` matches exposure to the screenshot (log-average luminance, secant steps, best-of). Only structural changes (winding, material wiring) need `process`/`build` again.

## Unreal (ue/, `gtb.py unreal*`)
Tested with UE 5.8.3 on the same Frostpunk capture (2026-09-29): the game view is within 0.05 stops of Blender with both looks. The night look is within 0.03 stops of the screenshot, with no Unreal-specific calibration.
- Flow: `ue/export.py` (manifest → plan.json + glb chunks, no Unreal needed) → `ue/blender_lut.py` (Blender bakes grade + AgX into a LUT) → `ue/editor/gtb_ue.py`, run by `UnrealEditor-Cmd -run=pythonscript` with `-nullrhi` → Movie Render Queue in `-game -RenderOffscreen` → `unreal/parity.png` (game | Blender | Unreal). A full build takes about 90 s, a look pass about 40 s, and a render about 30 s.
- Material rules come from `gtb/scene_common.py` (shared with Blender). The master HLSL in `ue/editor/gtb_hlsl.py` mirrors the Blender node trees, and the snow noise is an exact port of Cycles' Perlin noise. Bump `MASTER_VERSION` in `gtb_materials.py` after changing a master.
- Colour: the LUT pass runs *before* bloom (BL_SceneColorBeforeBloom). It outputs linear values, divided by EyeAdaptation, through a neutral tonemapper (Filmic with tone curve 0). **Do not use BL_ReplacingTonemapper.** MRQ captures linear (SCS_FinalToneCurveHDR) and re-applies sRGB itself, so a display-encoded replacement comes out twice as bright.
- Units match Blender one to one: sun W/m² = lux, point light P W = P/(4π) cd, world colour = sky luminance. Height fog density = 10·d/ln(2)² with falloff ≈ 0 (from HeightFogCommon.ush).
- UE Python gotchas:
  - Commandlet prints don't reach stdout, so write a log file.
  - Use forward slashes in the `-script=` path, because `\u...` gets eaten.
  - A single-mesh glb is named after the file.
  - `spawn_actor_from_object` returns None in the commandlet, so spawn StaticMeshActor by class instead.
  - Coincident vertices make the mesh build quadratic.
  - Sequencer keys are in ticks (display frame × 1000 at 24 fps).
  - Git Bash rewrites `/Game/...` arguments; the pipeline calls Unreal from Python, so it isn't affected.
- Known difference: Lumen occludes sky light with off-screen geometry, which EEVEE's screen-space GI can't. So the close-ups are up to ~0.4 stops darker than Blender's, while the game view matches. `look.unreal.cvars` can change render cvars.

## Checks
`python tests/selftest.py` builds a synthetic rip and runs the whole pipeline (must print all PASS). It also checks the Unreal export (plan, glb, LUT) without Unreal.
