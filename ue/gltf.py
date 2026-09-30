"""Minimal glTF 2.0 binary (.glb) writer: geometry only, no materials.

Unreal's glTF importer converts glTF (x, y, z) to UE (x, z, y) and metres to
centimetres, so callers pass positions already in glTF axes.
"""
import json
import struct

import numpy as np

FLOAT, UINT = 5126, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


class GlbWriter:
    def __init__(self):
        self.bin = bytearray()
        self.doc = {"asset": {"version": "2.0", "generator": "FreezeFrame"},
                    "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [],
                    "accessors": [], "bufferViews": [], "buffers": []}

    def _view(self, data: bytes, target):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.doc["bufferViews"].append({"buffer": 0, "byteOffset": len(self.bin),
                                        "byteLength": len(data), "target": target})
        self.bin += data
        return len(self.doc["bufferViews"]) - 1

    def _accessor(self, arr, kind, target, minmax=False):
        arr = np.ascontiguousarray(arr)
        comp = UINT if arr.dtype == np.uint32 else FLOAT
        acc = {"bufferView": self._view(arr.tobytes(), target), "componentType": comp,
               "count": int(arr.shape[0]), "type": kind}
        if minmax:
            acc["min"] = arr.min(0).tolist()
            acc["max"] = arr.max(0).tolist()
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1

    def add_mesh(self, name, positions, indices, normals=None, uvs=(), colors=None):
        """positions/normals (N,3), indices (T,3), uvs: list of (N,2), colors (N,4); glTF axes."""
        f32 = lambda a: np.asarray(a, dtype=np.float32)
        attrs = {"POSITION": self._accessor(f32(positions), "VEC3", ARRAY_BUFFER, minmax=True)}
        if normals is not None:
            attrs["NORMAL"] = self._accessor(f32(normals), "VEC3", ARRAY_BUFFER)
        for i, uv in enumerate(uvs):
            attrs[f"TEXCOORD_{i}"] = self._accessor(f32(uv), "VEC2", ARRAY_BUFFER)
        if colors is not None:
            attrs["COLOR_0"] = self._accessor(f32(colors), "VEC4", ARRAY_BUFFER)
        idx = self._accessor(np.asarray(indices, dtype=np.uint32).ravel(), "SCALAR", ELEMENT_ARRAY_BUFFER)
        self.doc["meshes"].append({"name": name, "primitives": [{"attributes": attrs, "indices": idx, "mode": 4}]})
        self.doc["nodes"].append({"name": name, "mesh": len(self.doc["meshes"]) - 1})
        self.doc["scenes"][0]["nodes"].append(len(self.doc["nodes"]) - 1)

    def save(self, path):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.doc["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(self.doc, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(self.bin)))
            f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942) + bytes(self.bin))

    def __len__(self):
        return len(self.doc["meshes"])


def blender_to_gltf(v):
    """Blender (x, y, z) -> glTF (x, z, -y); UE then reads it as (x, -y, z)."""
    v = np.asarray(v)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], -1)
