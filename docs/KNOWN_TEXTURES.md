# The known-texture lookup table

This note is for a Claude session (or a person) working on a Frostpunk capture made with Ninja Ripper. It explains what `profiles/frostpunk_textures.json` is, why it helps every future capture, and how to read it, add to it and override it.

**In short: it is a lookup table, texture hash → what that texture is. Its only job is to get a scene built faster, by reusing answers that were already worked out instead of guessing again. It is not a rulebook. If an entry is wrong, or wrong for your scene, override it.**

## What it is

A rip doesn't say what a texture is for. The game only binds textures to numbered slots, so the pipeline has to work out which one is the colour (albedo), which is the normal map, and which is a mask. It does that from the pixels and from how the shaders use the slots. Those rules are good, but they still guess, and a wrong guess is very visible: a normal map used as colour paints a building green and orange, and an atlas used as colour paints trees with a moss tile.

The lookup table skips the guess for every texture someone has already worked out. It starts with 170 textures of the capture `Frostpunk_20260929_115750`, whose materials were audited one by one. Each entry has:

- `hashes`: a hash of the raw bytes of each mip level of the texture (16 px and up).
- `role`: what the texture is: `albedo`, `normal`, `gray`, `packed`, `shared` or `detail`. A normal map also has `channels` (`AG`, `RG` or `RGB`).
- `what`: a short note, for the textures that were identified (coal, banner cloth, the rock and moss atlas, and so on).
- `ref` and `materials`: the capture and texture id where it was checked, and which Unreal materials use it there.

When a rip is processed, every texture whose hash is in the table gets the role stored there instead of a guess. Nothing else reads the table: it doesn't change meshes, lights or the look, and deleting an entry just sends that texture back to the rules.

## Why it helps a capture made on another PC

The game ships the same texture files to everyone, so the same texture is the same bytes in every rip, on any PC. That was tested on 2026-09-30 with a rip from a second PC, taken at the start of a game: 55 of its 85 textures matched the table byte for byte, including the terrain, the crater walls, the rocks, the smoke, fire and mist flipbooks and the lamp glows. The textures that didn't match were early-game buildings that the reference capture doesn't have.

Two details make the match reliable:

- **Each mip level is hashed separately.** The game streams textures, so one rip can hold a texture from 2048 px down and another from 1024 px down. The levels they share are the same bytes, so they still match.
- **Single-colour textures are skipped.** Two flat textures of the same colour are the same bytes whatever they are used for, so a match would say nothing about the role. Cube maps are skipped too: their role comes from the file header and is never a guess.

Every texture that gets checked and saved helps all later captures, from either PC. The more captures are checked, the fewer guesses are left.

## How to use it

It needs no setup: `profiles/frostpunk.json` names the table (`known_textures`), and `process` applies it to every new rip. It prints a line like:

```
55 of 70 textures match the known-texture database (frostpunk_textures.json): they get its checked roles. The other 15 use the rules; check them with `python gtb.py audit <capture>`
```

For a capture that was processed before the table existed, run these (the rip's `.dds` files must still be where `capture.json` points):

```bash
python gtb.py textures <capture>
```

```bash
python gtb.py audit <capture>
```

```bash
python -m ue.live materials <capture>
```

The first applies the table and the rules again. The second writes `captures/<capture>/audit/`: sheets with one row per material, showing each texture, its role and a crop of the game screenshot where that material appears, plus a `report.md`. Textures marked **K** came from the table; the ones marked **?** were guessed and are listed first. The third pushes the materials into the open Unreal editor.

Look at the **?** rows on the sheets before asking the user what looks wrong. A normal map used as colour shows green, orange or blue streaks. An atlas shows a patchwork. A mask shows black and white. The report also says which material of the reference capture uses the same colour texture, because material numbers (`MI_M###`) are different in every rip.

## How to save and override values

**Save everything a capture has, without checking each texture:**

```bash
python gtb.py textures <capture> --save-known
```

Do this once a capture is built. The rule is simple: if nobody says a texture looks weird, what is stored counts as correct. There is no need to inspect every texture first.

**When a texture does look wrong, fix its entry.** Saving and overriding are the same command: a texture that is already in the table gets its entry replaced. Each entry stands for one texture, and entries don't refer to each other, so:

- One texture with the wrong role (say, a normal map stored as a colour) means changing one entry. The material then rebuilds itself from the roles.
- Two textures whose roles were swapped (the colour stored as the normal, and the normal as the colour) means changing both.

One texture (new, or to override the entry it has):

```bash
python gtb.py known <capture> t0058 --role albedo --what "early shelters: bricks, planks and metal"
```

A normal map needs its channel layout: `--role normal:AG` (Frostpunk packs most normals in alpha and green), `normal:RG` or `normal:RGB`.

To see what's there:

```bash
python gtb.py known <capture>
```

```bash
python gtb.py known <capture> t0058
```

The first lists the capture's textures that aren't in the table. The second shows one texture's role and its entry.

After saving or overriding, `python gtb.py textures <capture>` applies the roles to the capture, and `python -m ue.live materials <capture>` pushes them into the open Unreal editor.

To drop an entry and let the rules decide again, delete it from the JSON file (it is plain text, one block per texture).

From Python, the same things are `gtb.textures.known_get(db, texture)` and `gtb.textures.known_add(db, texture, role, channels, what, ref)`. `db` is the table's file or the game profile; `texture` is a `.dds` file or a list of its mip hashes (a manifest texture entry's `hashes`). Adding a texture that is already there replaces its entry (that is how you override); it never makes a second one.

## Sharing it between two people

The table is one file, `profiles/frostpunk_textures.json`. To combine someone else's copy with yours, don't overwrite yours. Merge it:

```bash
python gtb.py known <capture> --merge path/to/their/frostpunk_textures.json
```

Textures you don't have are added. For textures you both have, their role is taken and your note stays. Running it twice changes nothing.

## Limits

- It is a shortcut, so it can be out of date or wrong for a scene; override it when it is.
- It knows only what has been saved. A building that isn't in any checked capture still gets guessed roles; that is what the audit is for.
- It matches exact bytes. A game update that changes a texture, or a Ninja Ripper setting that saves textures in another format, gives different hashes. Then the texture is simply treated as new.
- Textures the game draws into at run time (render targets, the generator's smoke) are different in every frame and never match.
