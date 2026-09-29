"""blender -b <capture>/scene.blend --python closeups.py -- <capture_dir> <out_prefix>
Renders close-up checks: the game camera dollied toward what it looks at
(35% and 15% of the distance), so lighting is judged up close too."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402
import gtb_scene  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
capture, prefix = Path(args[0]).resolve(), args[1]
scene = bpy.context.scene
gtb_scene.apply_look(scene, gtb_scene.load_look(capture / "look.json"))
cam = scene.camera
origin = cam.matrix_world.translation.copy()
fwd = (cam.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
hit, target, *_ = scene.ray_cast(bpy.context.evaluated_depsgraph_get(), origin, fwd)
if not hit:
    target = origin + fwd * 100
scene.render.resolution_percentage = 40
import math  # noqa: E402
views = [("35%", target + (origin - target) * 0.35, None), ("15%", target + (origin - target) * 0.15, None)]
# Low angle: same heading, 20 deg above the ground, at 15% of the distance.
dist = (origin - target).length * 0.15
flat = Vector((fwd.x, fwd.y, 0)).normalized()
low = target - flat * dist * math.cos(math.radians(20)) + Vector((0, 0, dist * math.sin(math.radians(20))))
views.append(("low", low, (target - low).to_track_quat("-Z", "Y")))
rot0 = cam.matrix_world.to_quaternion()
for i, (name, pos, rot) in enumerate(views):
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = rot or rot0
    cam.location = pos
    scene.render.filepath = f"{prefix}_{i}.png"
    bpy.ops.render.render(write_still=True)
    print(f"[gtb] closeup {name} -> {scene.render.filepath}")
