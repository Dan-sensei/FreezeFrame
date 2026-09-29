"""blender -b <capture>/scene.blend --python render.py -- <capture_dir> <out.png> [--save]
Re-applies look.json to the saved scene and renders the game camera at
look.render.preview_scale. --save also writes the updated look back into scene.blend."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bpy  # noqa: E402
import gtb_scene  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
capture, out = Path(args[0]).resolve(), Path(args[1]).resolve()
scene = bpy.context.scene
look = gtb_scene.load_look(capture / "look.json")
gtb_scene.apply_look(scene, look)
if "--save" in args:
    bpy.ops.wm.save_mainfile()

scene.render.resolution_percentage = int(round(look["render"]["preview_scale"] * 100))
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(out)
bpy.ops.render.render(write_still=True)
print(f"[gtb] rendered {out}")
