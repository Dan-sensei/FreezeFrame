# Generator smoke in Unreal (Niagara Fluids)

The generator's smoke column is drawn in the game from a live render target, which Ninja Ripper can't save, so the rip has no smoke there. In Unreal it is rebuilt as a real fluid simulation (Niagara Fluids): a column of light, bluish-grey smoke that rises out of the furnace top, billows, spreads and thins out as it climbs, with a warm orange glow at its base. It replaces the sprite column (`SmokeColumn00`, `M_GTB_Plume`), which the user rejected.

Tested on the Frostpunk capture `Frostpunk_20260929_115750` with UE 5.8.3 (2026-09-29). The look was tuned by eye with the user watching the editor viewport. The colour was then matched to the chimney smoke by measuring a screenshot.

## Two versions: offer the choice

There are two ways to show the generator's smoke, and both stay in the project. **Ask the user which one they want; never choose for them.** Ask when you recreate the scene (`gtb.py unreal` and `gtb.py unreal-open` end with a `GENERATOR SMOKE - ask the user ...` reminder), or when they work on the generator. Describe both, as in the table, and mention that switching takes a second and rebuilds nothing. `CLAUDE.md` ("Ask the user first") has a suggested wording.

| version | what it looks like | made by |
|---|---|---|
| **fluid** (default, approved by the original user) | A real simulation: a light, bluish-white column with billows. It rises out of the furnace top, spreads, and thins and lightens as it climbs, with an orange glow at its base. It costs GPU time and takes ~30 s to fill after a restart. | `python -m ue.fluid_smoke <capture>` in the open editor |
| **sprites** | Camera-facing puffs from the game's own smoke flipbook, rising in a loop: a darker, denser column, lit by the fire from below, with flames at its base (modelled on the game's burning-generator art). Cheap, and it animates in renders at once. | `python gtb.py unreal <capture>` (the level build), always |

```bash
python -m ue.fluid_smoke Frostpunk_20260929_115750 --show sprites
```

```bash
python -m ue.fluid_smoke Frostpunk_20260929_115750 --show fluid
```

`--show` only changes visibility, as one undo step, unsaved. It pauses the simulation while the sprites show. Every run of the script prints the command to switch. One exception: if `GeneratorPlume` has lost its Niagara system (see Pitfalls, "A plume that shows nothing"), `--show fluid` rebuilds the plume instead.

## Recreate it

You need:

- **The capture's Unreal level**, built by `python gtb.py unreal <capture>` (see [RECREATE.md](RECREATE.md), step 4).
- **The editor open on that level**: `python gtb.py unreal-open <capture>`.
- **Three plugins:** the project template ([ue/project_template](../ue/project_template)) enables NiagaraFluids, ToolsetRegistry and NiagaraToolsets. `gtb.py` copies the template into `unreal_project/`. If the editor was already open when the plugins were added, restart it once.
- **Python Remote Execution**, which the template's `DefaultEngine.ini` turns on. That file also holds the Heterogeneous Volume settings below.

Then, with the editor open:

```bash
python -m ue.fluid_smoke Frostpunk_20260929_115750
```

The script runs in about a minute: it copies and compiles the material, then copies and recompiles the Niagara system. It prints `GTB fluid smoke: GeneratorPlume at (-5031, -8483, -12311), M_GTB_FluidGas rebuilt, ...`. The simulation starts empty and fills in about 30 s; an editor that isn't in focus ticks slower, so allow 60–90 s there. The script saves the three assets it makes (`M_GTB_FluidGas`, `MI_GTB_GeneratorPlume`, `NS_GTB_GeneratorPlume`) right away. The level changes (the actors, the hidden sprite column) are one undo step and **unsaved**: File → Save All keeps them.

To try changes without touching the real plume, build a copy beside it and delete it afterwards:

```bash
python -m ue.fluid_smoke Frostpunk_20260929_115750 --suffix _Test --offset 0 1500 0
```

```bash
python -m ue.fluid_smoke Frostpunk_20260929_115750 --remove --suffix _Test
```

## What the script builds

| what | where | notes |
|---|---|---|
| `M_GTB_FluidGas` | `/Game/GTB/Shared` | A copy of NiagaraFluids' `M_3DGas_Base` (the volume material the gas renderer uses), with `GTB_*` controls spliced in; see below. Rebuilt when `MASTER_VERSION` in `ue/fluid_smoke.py` changes. |
| `MI_GTB_GeneratorPlume` | `/Game/GTB/Shared` | The look: every `GTB_*` value comes from the settings. |
| `NS_GTB_GeneratorPlume` | `/Game/GTB/Shared` | A copy of `/NiagaraFluids/Templates/Gas/3D/Systems/Grid3D_Gas_Fire`. The source emitter's density and temperature multipliers are raised through the NiagaraToolsets plugin, and the volume renderer's `material` is set to the MI. |
| `GeneratorPlume` | level, folder `Smoke` | The Niagara actor, 1.5 m inside the furnace, scale 2. User parameters `WorldSpaceSize` and `ResolutionMaxAxis`. |
| `GeneratorFireLight` | level, folder `Smoke` | An orange point light 0.5 m above the rim, with a short reach: the warm base. |
| `SkyDome` | level | Collision profile set to `NoCollision` (see "Pitfalls"). |
| `SmokeColumn00` | level | Hidden, not deleted: the sprite column this replaces. |

Where it goes: the capture's plan has one plume source (`plan.json` → `plume.plumes[0].location`, the lowest puff of the game's smoke column). The furnace rim is 760 cm above it (`rim_above_source`, measured by eye on this capture). The furnace centre is 8 cm toward +x and 71 cm toward +y from it (`center_offset`); at the plume source the smoke visibly rose off-centre. The offset was measured two ways: a circle fitted to the cap's orange ring in a straight-down capture with the plume hidden (radius 2.96 m), then side captures at rim height comparing the column with the red drum. The source itself is shifted 13 cm toward −x (`source_shift`), because the template's `FireSourceMesh` is lopsided: its surface centroid is 6.5 cm off-centre in local x. The simulation grid is centred on the actor.

### The material controls (`M_GTB_FluidGas`)

Niagara drives the original material parameters every frame through the renderer's bindings, so they can't be changed from a material instance. The copy keeps those bindings and adds controls after them:

| parameter | what it does |
|---|---|
| `GTB_Density` | Extinction (opacity) multiplier. |
| `GTB_DensityBind` | 1 = multiply Niagara's `DensityGain`; 0 = ignore it. The fire template binds it to 0, so its smoke never shows. |
| `GTB_SmokeAlbedo`, `GTB_AlbedoBind` | The smoke colour. With bind 0 it replaces Niagara's colour, which is black in the fire template. |
| `GTB_BaseZ`, `GTB_FadeHeight` | h = saturate((world z − BaseZ) / FadeHeight): 0 at the rim, 1 at FadeHeight above it. |
| `GTB_TopDensity`, `GTB_TopAlbedo` | Density factor and colour at h = 1: the smoke gets thinner and lighter as it rises. |
| `GTB_TopGlow` | A little emission that grows with h, lifting the upper smoke. Above ~0.5 it turns flat white. |
| `GTB_FireGain`, `GTB_EmissiveBind` | Scale the template's fire emission (the orange heat glow) and all of Niagara's emission. |

## Settings

`look.unreal.fluid_smoke` in the capture's `look.json`, over `UNREAL_DEFAULTS["fluid_smoke"]` in [ue/look.py](../ue/look.py). These are the values the user approved:

| key | value | effect |
|---|---|---|
| `rim_above_source` | 760 | cm from the plan's plume source to the furnace rim |
| `center_offset` | 8, 71 | cm (x, y) from the plan's plume source to the furnace centre (the light goes here) |
| `source_shift` | −13, 0 | cm (x, y) added to the plume actor only: compensates the lopsided source mesh |
| `depth` | 150 | cm the source sits below the rim (the smoke comes out of the furnace, not from above it) |
| `scale` | 2 | actor scale; the source (the template's `FireSourceMesh`, 1 m × 0.8 m) scales with it |
| `grid` | 700, 700, 2400 | `WorldSpaceSize` in local units, so 14 × 14 × 48 m. Centred on the actor, so its top is 22.5 m above the rim. Too small a grid shows as a hard wall or a flat top. |
| `resolution` | 320 | `ResolutionMaxAxis`. 256 also works; 384 didn't come up in the editor. |
| `source_density` | 9 | multiplier on the source particles' density (fire template default 1) |
| `source_temperature` | 2.5 | multiplier on their temperature: more buoyancy, so it rises faster and higher |
| `density` | 1.2 | `GTB_Density` (overall opacity) |
| `albedo` | 0.485, 0.727, 1.0 | smoke colour, matched to the chimney smoke sprites |
| `top_albedo` | 0.567, 0.787, 1.0 | colour at the top of the fade |
| `top_density` | 0.25 | density factor at the top of the fade |
| `fade_height` | 2800 | cm above the rim over which it thins and lightens |
| `top_glow` | 0.25 | `GTB_TopGlow` |
| `fire_gain` | 0.02 | the template's heat glow, kept faint |
| `light_cd`, `light_depth`, `light_reach`, `light_radius`, `light_color` | 150, −50, 900, 150, (1, 0.45, 0.15) | the orange base light: candela, cm below the rim (negative = above), attenuation radius, source radius, colour |

Heterogeneous Volume console variables (in `DefaultEngine.ini`, `[SystemSettings]`; the script also sets them live):

| cvar | value | why |
|---|---|---|
| `r.HeterogeneousVolumes.IndirectLighting` | 0.35 | Sky light on the volume. At 0 the smoke only sees the sun and is black wherever it shadows itself; at 1 the billows lose their shading. |
| `r.HeterogeneousVolumes.IndirectLighting.Mode` | 2 | single scattering |
| `r.HeterogeneousVolumes.SupportOverlappingVolumes` | 1 | Without it, overlapping volumes swap draw order from frame to frame. |

## The sprite column (the other version)

- **Built by the level build:** `gtb.py unreal` places `SmokeColumn00` (tag `gtb_plume`, folder `Smoke`) at each plume source in the plan. It uses the static mesh `SM_Plume`: 320 puff quads written by `ue/export.py` `write_plume`, from the game's smoke flipbook with the most frames (`t0121`, 8 × 8).
- **The shader:** the `M_GTB_Plume` master (`gtb_hlsl.PLUME`, `gtb_materials.build_plume`) moves the puffs up the column over time and faces them to the camera. Its fBm noise erodes each puff into a crisp billow. The fire at the vent lights the side facing it (`FireReach`), the rest is charcoal (`ShadowColor` → `SmokeColor`), and the youngest puffs are flames (`FlameHeight`, `FlameStrength`).
- **Its settings:** `look.unreal.plume` in `look.json`, over `UNREAL_DEFAULTS["plume"]` in `ue/look.py`. The keys are `enabled`, `speed`, `glow`, `wind_drift`, `detail`, `shadow`, `opacity`, `puff_size`, `fire_reach`, `flame`, `flame_height` and `spread`, described in the docstring at the top of `ue/look.py`. Height, growth, colour and density come from `look.smoke`, as for Blender's volumetric plume.
- **To push changes into the open editor:** `python -m ue.live plume <capture>` rebuilds `M_GTB_Plume`, re-imports `SM_Plume` and sets `MI_Look_Plume` from `look.json`. Show it first with `--show sprites`.
- **Why the fluid is the default:** the original user rejected this column, even after the rework, as not looking like real smoke, and chose the Niagara Fluids version.

## The user's look (what was asked for)

- The smoke rises from the furnace top as a column, not a cloud that envelops the area. Too wide was the first complaint.
- It has texture (visible billows) and doesn't read like the soft sprite smoke around it.
- It rises high and spreads, getting a little lighter and more transparent as it climbs. A strong white glow was rejected.
- The colour matches the chimney smoke: bluish white. Measured in linear RGB on the user's screenshot, the chimney smoke has red/green 0.74 and blue/green 1.37.
- Orange at the base only: the fire is a detail inside the furnace, and visible flames were rejected ("random fire").
- No tilt, no sideways drift, no visible grid edge.

## Pitfalls (what went wrong on the way)

- **Use the fire template, not the smoke template.** `Grid3D_Gas_Smoke` has a turbulence bias of (−0.45, 0, 0) in world space baked into a simulation stage, so its smoke always drifts toward −x and piles against the grid wall. Rotating or tilting the actor doesn't cancel it. Simulation-stage module inputs are out of reach from Python: NiagaraToolsets looks stages up with an empty GUID. `Grid3D_Gas_Fire` has no bias, and its source is an ordinary particle emitter (`ParticleSourceEmitter`) whose inputs the toolset can set.
- **Sky dome collision.** The sky dome is the engine sphere scaled to 5 km. With collision, the fluid treats the whole city as inside a solid and shows nothing. `set_collision_enabled(NO_COLLISION)` alone reverts when the level loads, because the `BlockAll` profile re-applies itself. Set the profile to `NoCollision`; `gtb_ue.py` and the script both do.
- **One volume, not two.** A separate fire system and smoke system overlapping at the furnace flickered, with the fire jumping in front of the smoke. One system's own heat glow is the fire.
- **Settings that make the volume vanish:** `r.HeterogeneousVolumes.DownsampleFactor 1`, and the volume renderer's `LightingDownsampleFactor 1`. Keep both at 2.
- **The source can't be offset inside the grid.** `StaticMeshLocation` → `Sampled Position Offset` had no effect, so the grid stays centred on the actor and half of it is below the furnace.
- **Orange everywhere.** The fire template gives the gas an orange colour and emission; `GTB_AlbedoBind 0` and a low `GTB_FireGain` handle that. The base light's reach must stay short: at 15 m it tinted the whole plume peach.
- **A plume that shows nothing after reopening the level.** Its Niagara component has no system (`get_asset()` is None): the level was saved with the `GeneratorPlume` actor, but not the new `NS_GTB_GeneratorPlume` and `MI_GTB_GeneratorPlume` assets. Ctrl+S saves only the level, and closing the editor without saving then drops those unsaved assets. That happened on 2026-09-30. The script now saves its own assets as soon as it builds them. `--show fluid` detects a plume without a system and rebuilds it (about 20 s). Running `python -m ue.fluid_smoke <capture>` fixes it too.
- **Actor moves can silently not stick.** Read the location back after moving it.
- **Captures.** A `SceneCapture2D` renders the fluid, but the background editor ticks slowly: wait 60–90 s after `reset_system` before capturing. The user, watching the focused viewport, sees it in 30 s, so for colour and look tweaks let them judge in the viewport.

## Python routes that work (UE 5.8)

- **Module inputs:** `unreal.ToolsetRegistry.execute_tool("NiagaraToolsets.NiagaraToolset_System", "SetStackInputData", json)`. The stack reference is `{system: {refPath}, emitterName, scriptName, moduleName, rendererIndex: -1, inputNameStack: [...]}`, and the value `{struct: {refPath: "/Script/Niagara.NiagaraFloat"}, value: {value: 9.0}}`. `GetEmitterTopology`, `GetEmitterInputValues` and `GetDynamicInputChain` list what's there. They work for particle emitters, not for the gas master emitter or simulation stages.
- **Renderer material:** `unreal.find_object(None, "<system path>:Grid3D_Gas_Master_Emitter.NiagaraVolumeRendererProperties_0")`, then `set_editor_property("material", mi)`.
- **Hidden properties:** `unreal.ToolsetLibrary.get_object_properties` / `set_object_properties` read and write UPROPERTYs that `get_editor_property` refuses, such as `LightingDownsampleFactor`.
- **User parameters:** `NiagaraComponent.set_variable_vec3/int/bool("WorldSpaceSize", ...)`, with no `User.` prefix. Spawn the actor with `spawn_actor_from_object(system, location)`; an empty NiagaraActor followed by `set_asset` stays inactive.

## Not done yet

- `gtb.py unreal` (the headless build) doesn't create the fluid plume: run `python -m ue.fluid_smoke <capture>` in the editor after each full build. The build always makes the sprite column; the script hides it, and `--show` switches between them.
- Movie Render Queue renders (`gtb.py unreal-render`) are untested with the fluid. It needs time to fill: use the component's warm-up (`override_warmup_settings`, `warmup_tick_count`) or a longer MRQ warm-up.
- Blender has no equivalent (its volumetric plume, `smoke.enabled`, is off).
- The hidden `GeneratorSmoke` / `GeneratorFire` actors and the `NS_GTB_GeneratorSmoke` / `MI_GTB_GeneratorSmoke` assets are leftovers from the smoke-template attempt and can be deleted.
