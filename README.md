# GameToBlender

Press a key in a game and get a Blender scene of that frame. The scene has the same camera, the game's textures wired into PBR materials, and lighting, fog and grading that are matched against a screenshot taken at the same moment.

```
game ──(Ninja Ripper frame rip + our screenshot, same keypress)──▶ captures/<game>_<time>/
  process  : .nr draw calls → meshes (npz) + textures (png) + solved camera → manifest.json
  build    : Blender 5.2 headless → scene.blend (meshes, materials, camera, sun/sky/fog, compositor)
  calibrate: render → compare with screenshot → auto-match exposure
  tune     : edit look.json → `gtb.py render` → comparison.png (by hand, or ask Claude)
```

## One-time setup

0. Python 3.11+ and Blender 5.2+. Install packages: `python -m pip install -r requirements.txt`, then check with `python tests/selftest.py` (all PASS).

1. **Ninja Ripper 2.x.** Download it from ninjaripper.com (it's paid, through Patreon or Boosty) and extract it.
2. **Disable overlays**: Steam, GeForce/AMD, RTSS/Afterburner, OBS. Turn off DLSS/FSR/XeSS. Run the game **borderless windowed**, because screenshots of exclusive fullscreen can come out black.
3. **In the Ninja Ripper settings:**
   - Use a **frame rip** (PrintScreen by default).
   - Save textures as **DDS**.
   - Keep the **post-vertex-shader ("world space") geometry**, and the pre-VS (local) data too if there's an option for it. The pre-VS data lets the FOV be solved exactly. Without it the profile's `fov_y_deg` is used, which still lines up perfectly on screen but can be stretched in depth.
4. If `python gtb.py watch` reports the wrong Ninja Ripper output folder, set `ripper_output_dir` in `config.json`. It's normally read from Ninja Ripper's own settings.

## Use

1. Double-click `GameToBlender.bat`, or run the command below. It starts the capture daemon.

   ```bash
   python gtb.py watch
   ```

2. Launch the game **through Ninja Ripper**. For Steam games, fully exit Steam, start `steam.exe` from Ninja Ripper, then start the game.
3. Hide the HUD if you can, then press **PrintScreen**. The daemon screenshots that frame, waits for Ninja Ripper to finish writing, then runs process, build and calibrate.
4. Open the result, either `captures/<name>/scene.blend` directly or with:

   ```bash
   python gtb.py open latest
   ```

   The camera shows the game screenshot as a 50% overlay, so you can check the match.

Other commands: `process | build | render | calibrate | open | all <capture>`. `<capture>` is a folder under `captures/` or `latest`.

## The tuning pass

`captures/<name>/look.json` controls everything that isn't geometry. After editing it, re-render:

```bash
python gtb.py render latest --save
```

This writes `comparison.png`. The top row shows the game and the Blender render side by side. The bottom row shows the edge overlay (camera alignment) and a luminance difference (red = too bright, blue = too dark). The command also prints metrics: exposure offset, color balance, per-region brightness ratios and edge alignment.

| key | what it does |
|---|---|
| `exposure`, `view_transform`, `look` | Color management (AgX looks are named like `"AgX - Punchy"`) |
| `sun` | `azimuth`/`elevation` in degrees (azimuth is clockwise from +Y); plus `strength`, `color`, `angle` (softness) |
| `sky` | `type`: `MULTIPLE_SCATTERING`, `SINGLE_SCATTERING`, `PREETHAM`, `HOSEK_WILKIE` or `color`; plus `strength`, `color` |
| `fog` | Depth fog: `density`, `color`, `strength`, `start`, `max_opacity` |
| `bloom`, `grade` | Compositor bloom; saturation, contrast, lift/gamma/gain |
| `materials` | Global `albedo_gain`, `saturation`, `roughness`, `normal_strength`, `specular` |
| `lights` | Extra lights: `[{type, location, color, energy, radius}]`, e.g. Frostpunk's generator glow and street lamps |
| `compare_ignore` | Screen rectangles to leave out of the metrics, e.g. HUD: `[[x0,y0,x1,y1]]` in 0–1 fractions |

Or just ask Claude: *"tune the latest capture"*. It will iterate on `look.json` using the comparison sheets.

## Game profiles (`profiles/*.json`)

A profile is matched by executable name. Its keys:

- `fov_y_deg`: FOV fallback when it can't be solved.
- `min_up_dot`: ground detection. Use 0 for top-down cameras like Frostpunk.
- `uv_decode`: `auto`, `half`, or `scale:<k>` for games with fixed-point UVs.
- `normal_flip_green`: flips the normal map's green channel (DirectX vs OpenGL convention).
- `gray_is_gloss`: treats grayscale maps as gloss instead of roughness.
- `slot_roles`: per-slot overrides, e.g. `{"0":"albedo","1":"normal","2":"ignore"}`.
- `packed_channels`: for packed mask textures, e.g. `{"R":"Metallic","G":"1-Roughness"}`.
- `look`: the starting `look.json`.

The Frostpunk and Sekiro profiles are starting points. Refine them once a real rip shows which slot holds which texture.

## Checks after the first real rip

The console output of `process` tells you most of this:

- **"solved FOV from N rigid draws"**: pre-VS data is present, so the camera is exact. "FOV not solvable" means only post-VS data was ripped.
- **Skipped counts**: this line lists draws dropped as shadow maps, UI, duplicates and so on. If far too much is dropped, check `skip_offsize_rt` (games with dynamic resolution).
- **Texture roles**: albedo, normal, gray, packed. Wrong guesses are fixed with `slot_roles` in the profile.
- **UVs**: if textures look like noise, set `uv_decode` in the profile. Each mesh's vertex layout is recorded in `manifest.json` (`layout_pre`/`layout_post`).

## Known limits

- Game lights and shaders aren't in the rip, only geometry and textures. Lighting is rebuilt and matched, not extracted.
- Skinned characters, foliage with wind and GPU particles are placed correctly on screen, but they are frozen in their posed state.
- Everything is placed in camera space. The camera sits at the origin with its real pitch, and the ground is levelled to +Z.

## Self-test (no game needed)

```bash
python tests/selftest.py
```

This builds a synthetic Ninja Ripper rip with a known camera and junk draws, then runs the whole pipeline and checks the results.
