"""meshybench: the three published benchmarks for image-to-3D generation.

    meshybench.meshalign      geometry alignment against a reference mesh
    meshybench.texalign       texture alignment against a reference and Mesh-Texture Agreement
    meshybench.mesh_details   detail richness and detail quality, reference free
"""
from . import mesh_details, meshalign, texalign

__version__ = "1.0.0"
__all__ = ["meshalign", "texalign", "mesh_details", "__version__"]
