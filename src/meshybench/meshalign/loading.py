"""Mesh loading: every triangle geometry of a file, node transforms applied, as one mesh."""
from __future__ import annotations

import trimesh


def load_mesh(path: str) -> trimesh.Trimesh:
    """Raises ValueError when the file holds no triangles or no surface area. trimesh decodes a
    Draco-compressed GLB to zero-area geometry, so such files must be decompressed first."""
    sc = trimesh.load(path, force="scene")
    parts = []
    for name, g in sc.geometry.items():
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        nodes = sc.graph.geometry_nodes.get(name, [])
        if nodes:
            for node in nodes:
                T = sc.graph.get(node)[0]
                gg = g.copy()
                gg.apply_transform(T)
                parts.append(gg)
        else:
            parts.append(g.copy())
    if not parts:
        raise ValueError(f"no triangle geometry found in {path}")
    mesh = trimesh.util.concatenate(parts)
    if float(mesh.area_faces.sum()) <= 0.0:
        raise ValueError(f"degenerate geometry in {path}: total surface area is 0 "
                         "(a Draco-compressed GLB must be decompressed first)")
    return mesh
