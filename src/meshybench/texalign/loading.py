"""Scene loading for texture alignment: every triangle geometry with its node transform applied,
materials and texture coordinates kept per geometry."""
from __future__ import annotations

import trimesh


def load_scene_geoms(glb_path: str) -> list:
    sc = trimesh.load(glb_path, process=False)
    if isinstance(sc, trimesh.Trimesh):
        return [sc]
    out = []
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]
        g = sc.geometry[gname]
        if g.faces is None or len(g.faces) == 0:
            continue
        g = g.copy()
        g.apply_transform(T)
        out.append(g)
    if not out:
        raise ValueError(f"no triangle geometry found in {glb_path}")
    return out
