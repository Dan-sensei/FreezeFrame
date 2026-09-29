"""blender -b --python build.py -- <capture_dir>
Builds scene.blend inside the capture folder from manifest.json + look.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bpy  # noqa: E402
import gtb_scene  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
capture = Path(args[0]).resolve()
out = capture / (args[1] if len(args) > 1 else "scene.blend")
gtb_scene.build(capture)
bpy.ops.wm.save_as_mainfile(filepath=str(out))
print(f"[gtb] saved {out}")
