"""blender -b --factory-startup --python ue/blender_lut.py -- <look.json> <out.png> [size] [lo] [hi]

Bakes Blender's colour pipeline for a look into a 3D LUT so Unreal can show
exactly the same colours: compositor grade (lift/gamma/gain, saturation,
brightness/contrast) -> view exposure/gamma -> view transform (AgX) + look.
Fog and bloom are spatial, so Unreal does those itself.

LUT layout (16-bit PNG, display-referred sRGB): width size*size, height size.
Pixel (x, y) holds the input colour (r, g, b) = decode(x % size, y, x // size),
decode(i) = 2 ** (lo + (hi - lo) * i / (size - 1)) scene-linear, row 0 at top.
"""
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "blender"))
import gtb_scene  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
look_path, out = Path(args[0]), Path(args[1])
size = int(args[2]) if len(args) > 2 else 64
lo = float(args[3]) if len(args) > 3 else -12.0
hi = float(args[4]) if len(args) > 4 else 8.0

look = gtb_scene.load_look(look_path)
look = gtb_scene.deep_merge(look, {"fog": {"density": 0.0}, "bloom": {"strength": 0.0}})

scene = gtb_scene.clear_scene()
w, h = size * size, size
scene.render.resolution_x, scene.render.resolution_y = w, h
scene.render.resolution_percentage = 100
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 1
scene.render.use_compositing = True
for attr, value in (("compositor_device", "CPU"), ("compositor_precision", "FULL")):
    if hasattr(scene.render, attr):
        setattr(scene.render, attr, value)
vs = scene.view_settings
vs.view_transform = look["view_transform"]
try:
    vs.look = look["look"]
except TypeError:
    vs.look = "None"
vs.exposure = look["exposure"]
vs.gamma = look["gamma"]

# Input grid: scene-linear values, log2-spaced per channel.
levels = 2.0 ** (lo + (hi - lo) * np.arange(size) / (size - 1))
x = np.arange(w)
r = levels[x % size][None, :].repeat(h, 0)
b = levels[x // size][None, :].repeat(h, 0)
g = levels[np.arange(h)][:, None].repeat(w, 1)       # g index = row from the top
rgba = np.stack([r, g, b, np.ones_like(r)], -1)[::-1]  # Blender pixel rows start at the bottom
img = bpy.data.images.new("gtb_lut_in", w, h, alpha=False, float_buffer=True)
img.pixels.foreach_set(rgba.astype(np.float32).ravel())

# Same grade chain as the Blender scene, fed from the grid instead of the render.
gtb_scene.apply_compositor(scene, look)
ng = scene.compositing_node_group
rl = next(n for n in ng.nodes if n.bl_idname == "CompositorNodeRLayers")
src = ng.nodes.new("CompositorNodeImage")
src.image = img
for link in list(rl.outputs["Image"].links):
    ng.links.new(src.outputs["Image"], link.to_socket)
ng.nodes.remove(rl)

scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "16"
bpy.ops.render.render()
out.parent.mkdir(parents=True, exist_ok=True)
bpy.data.images["Render Result"].save_render(str(out), scene=scene)
print(f"[gtb] LUT {size}^3 ({lo:+g}..{hi:+g} stops) -> {out}")
