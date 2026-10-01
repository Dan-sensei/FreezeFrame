"""blender -b --factory-startup --python fbx_clip.py -- <clip.fbx> <out.npz>
Reads an animation clip (e.g. a Mixamo download) and writes its skeleton, frame by
frame, for gtb/clips.py: per bone its name, parent, rest and posed head position and
rotation in world space (Blender axes, metres), and the clip's frame rate."""
import sys

import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1:]
src, out = args[0], args[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=src, ignore_leaf_bones=False, automatic_bone_orientation=False)
arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
scene = bpy.context.scene
act = arm.animation_data.action if arm.animation_data else None
f0, f1 = (int(round(v)) for v in act.frame_range) if act else (1, 1)
fps = scene.render.fps / scene.render.fps_base
names = [b.name for b in arm.data.bones]
parents = [names.index(b.parent.name) if b.parent else -1 for b in arm.data.bones]
W = arm.matrix_world


def rot(M):
    return np.array(M.decompose()[1].to_matrix())


rest_head = np.array([W @ b.head_local for b in arm.data.bones])
rest_rot = np.array([rot(W @ b.matrix_local) for b in arm.data.bones])
heads, rots = [], []
for f in range(f0, f1 + 1):
    scene.frame_set(f)
    heads.append([W @ pb.head for pb in (arm.pose.bones[n] for n in names)])
    rots.append([rot(W @ pb.matrix) for pb in (arm.pose.bones[n] for n in names)])
np.savez_compressed(out, names=np.array(names), parents=np.array(parents), rest_head=rest_head,
                    rest_rot=rest_rot, head=np.array(heads), rot=np.array(rots), fps=fps)
print(f"[clip] {src}: {len(names)} bones, frames {f0}-{f1} at {fps:g} fps -> {out}")
