# GameToBlender: notes for Claude

Pipeline: Ninja Ripper 2 frame rip of a DX11 game → `gtb/process.py` (parse .nr, solve camera, classify draws/textures) → `blender/gtb_scene.py` (Blender 5.2 scene) → tune `look.json` against the game screenshot. See README.md for user-facing usage.

Tested end to end on **Frostpunk 1** with Ninja Ripper 2.18, Blender 5.2 and Unreal Engine 5.8.3 (2026-09-29). Sekiro has a starting profile but no real capture yet.

**To recreate the Frostpunk scene from the rip (Blender and Unreal), follow [docs/RECREATE.md](docs/RECREATE.md).** It covers `gtb.py import` → `all` → `unreal`, and lists the expected numbers at each step.

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
- Materials: albedo alpha is a **mask, not opacity**; normals packed in A+G (R ≈ roughness, except where R is 0 everywhere: that's the packer's fill, and as roughness it made the banners (`t0006`) perfect mirrors that showed the warm game lights as thin orange streaks. `scene_common.packed_channels` drops it, so those surfaces use `materials.roughness`); one engine-wide snow texture is bound to most shaders (auto-detected as "shared" and ignored). Meshes whose *only* texture is that one are snow drifts, which get a snow material. Terrain has no UVs and gets a flat colour. Shader families order slots differently, so there are no fixed slot numbers.
- Texture roles can't come from pixels alone. A packed A+G normal on a nearly flat surface has G and A pinned at 0.5 (`t0018`), and was taken for an albedo, painting the frosted planks, the banner stands and more green and orange. A frosted colour texture (`t0017`) is almost colourless and reads as grayscale. `classify` now has a flat-A+G rule, and `slot_consensus` settles albedo ↔ normal from the shaders: a pixel shader reads a slot the same way for every draw, so a texture that contradicts the role most textures in its (shader, slot) share gets that role (`t0091` → albedo, `t0159` → normal). Grayscale stays with `assign_slots` (terrain masks). On the Frostpunk capture this changes exactly 4 roles: `t0018`, `t0084` and `t0159` become normals, and `t0091` becomes an albedo. That fixes `MI_M008` (frosted planks: banner stands, towers, frames), `MI_M044`, `MI_M048` and `MI_M103`. `python gtb.py textures <capture>` re-runs the rules on a processed capture from its PNGs, without the rip; running it again changes nothing. Then `build` (it overwrites scene.blend, so ask first) and `unreal` pick the roles up, or `python -m ue.live materials <capture>` with the editor open.
- Some grayscale maps are masks for colours the shader computes, not base colours. One is the snow plateau's frost streaks (the only texture, mostly black). The others are cliff-top snow blends (bright, fading to black at the edges; `details.border_mean`). `assign_slots` marks them `mask`, so those surfaces get the ground/snow colour. Before this fix they rendered as black-and-white leopard spots.
- The game adds roof snow in-shader, so we use `add_surface_snow` (`materials.snow_cover` / `snow_threshold`).
- Banners (the red New London strips, 44 in the test capture) are found from the rip, not by name. The profile's `cloth` rule matches the banner shader pair, the game's vertex shader bending them (`rigid_fit` > 0.001, the pre-VS → post-VS affine fit error), and a hanging shape (height ≥ 2× width). One mesh on the same shader is a 10×9×4 m structure, and the shape test drops it. People and building parts are also non-rigid but use other shaders. The banners have no usable wind weight (`uv1` is constant (0,1) and colours are white), so the top of the bounds is the pinned edge. Each is a 4×9 vertex grid whatever its length (4–22 m).
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
  - The ObjectLocalBounds material node returns bounds padded by the material's Max World Position Offset Displacement (`PrimitiveSceneProxy::SetTransform`). Use PreSkinnedLocalBounds, which the engine un-pads. With 600 cm of padding, the cloth shader put each banner's top 6 m too high, so the top rows moved and the ripples came out about 1.75× too big. The numpy mirror couldn't see this, so check motion in Unreal too (per-pixel change over captured frames).
  - A VectorParameter's default output (`""`) is RGB only. Wire `"RGBA"` when the HLSL reads `.w`, or the shader fails with `vector swizzle 'w' is out of bounds`.
  - Git Bash rewrites `/Game/...` arguments; the pipeline calls Unreal from Python, so it isn't affected.
- Time: effects use `MPC_GTB_Time`. `SceneTime` is keyed by the sequences. `EngineTimeWeight` is 1 outside Sequencer, so the editor viewport animates. Anything that accumulates, like smoke rise, must use `SceneTime` only: the editor clock grows without limit, and sprites once ended up hundreds of metres up in the sky. Snowfall wraps, so engine time is safe for it.
- The user wants effects to animate in the editor viewport, not sit static. The capture's puffs loop (rise and fade) while `EngineTimeWeight` is 1, using a per-puff phase in UV6. The generator's smoke column (a live render target, not in the rip) is `M_GTB_Plume`: 128 puffs from the game's biggest smoke flipbook (`t0121`, 8x8), parameters from `look.smoke`.
- Banners (`M_GTB_Cloth`, `gtb_hlsl.CLOTH_WPO`) need no manual steps. `gtb.py unreal` builds them from a rip; `python -m ue.live cloth <capture>` pushes them into an open editor.
  - Settings: `look.unreal.cloth` over `UNREAL_DEFAULTS["cloth"]` in `ue/look.py`:
    - `ripple` 0.03 and `sway` 0.08: fractions of the banner's length at the snow's 2.5 m/s wind (`look.snow.wind` drives it).
    - `wavelength` 0.75: fraction of the length. Below ~0.6 the 9-row banners zigzag.
    - `speed` 0.5: ripples per second.
    - To tune live, change `Ripple`/`Sway`/`WaveLength`/`FlutterSpeed` on `MI_Look_Cloth`, then copy the values into look.json.
  - Motion: weight d² from the pinned top, so the upper part stays calm. A ripple along the banner's front direction travels down, and the whole strip swings downwind with slow gusts. Each banner has its own phase. The shader reads bounds from PreSkinnedLocalBounds (see the gotcha above).
  - Held banners: when the bottom edge has geometry within 15 cm on both sides (30 of 44, mostly because the bottom runs through the rigid stand's bottom block), the motion fades out over the lowest 35%, and there is no swing: on a banner fixed at both ends, a swing only bows the middle.
  - Room: the export's `cloth_room` casts horizontal rays in 32 directions from every vertex, plus distances to the slice's segment ends, so thin poles between rays count. It uses a 10 cm margin and allows 3 cm next to geometry a vertex already touches. The result is 8 direction budgets (cm per unit weight, each covering its 45° sector), stored as custom primitive data 0–11 (`ClothRoom` = front xy, bottom held, sides held; `ClothBudgetA/B`). Unreal's y flip turns direction k into −k. The shader fits each banner's motion to its budgets once, at the strongest gust: the ripple's amplitude and centre, the swing along the wall, and a last uniform scale for the diagonals.
  - Twins (front/back copies with opposite normals) share one canonical front direction; otherwise they pull apart.
  - Check: `python -m ue.cloth_check <capture>` replays the shader in numpy with the capture's look, in about 7 s. Expect `0 banner(s) clip, 0 jerky, 4 barely move`; the 4 are boxed in: 1003, 1005, 1014, 1022. With the limits removed, 41 clip. Keep `cloth_offsets` in sync with `CLOTH_WPO` and `export.cloth_weight`. Also look in Unreal (per-pixel change over captured frames), since the numpy mirror couldn't see the padded bounds.
  - Tried and dropped:
    - A fixed 6 m wavelength zigzagged on long banners.
    - A phase that varied across the width made the side edges fight.
    - Clamping each vertex per frame to the budget in its current direction snapped at sector boundaries (8–19 m/s spikes).
    - Front/back budgets from the inner strip only let an edge clip a beam.
    - 32 rays without the segment ends missed a pole.
    - A symmetric belly for held banners moved the upper half.
  - Blender has no cloth motion yet.
- Live edits in the user's open editor: `python -m ue.remote script.py` runs Python in the editor, which needs Project Settings → Plugins → Python → Enable Remote Execution. It sends the code as a temporary .py file, because the editor treats any text up to its first ".py" as a file name. `python -m ue.live cloth|materials <capture>` re-exports and pushes the banners, or every material instance and texture sRGB flag, into the capture's level, in about 15 s and 45 s. Wrap changes in `unreal.ScopedEditorTransaction` and leave saving to the user. After rebuilding a master, check it with `MaterialEditingLibrary.get_statistics`: it compiles synchronously and reports 0 instructions when the shader fails, and a failed master makes every mesh using it vanish. The log's "Shaders Compiled" line never appears when the shader comes from the cache. The editor doesn't render in the background, so viewport screenshots stay queued. To look at something, capture with a temporary SceneCapture2D (`capture_scene` + `RenderingLibrary.export_render_target`) and delete it afterwards.
- Never write to the Unreal project while the user has it open in the editor. `run_editor` refuses to, and for experiments you can point `cfg["unreal_project"]` at a scratch copy.
- Known difference: Lumen occludes sky light with off-screen geometry, which EEVEE's screen-space GI can't. So the close-ups are up to ~0.4 stops darker than Blender's, while the game view matches. `look.unreal.cvars` can change render cvars.

## Checks
`python tests/selftest.py` builds a synthetic rip and runs the whole pipeline (must print all PASS). It also checks the Unreal export (plan, glb, LUT) without Unreal.
