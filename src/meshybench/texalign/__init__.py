"""texalign: texture alignment of a textured model against its reference, and the reference-free
Mesh-Texture Agreement.

    from meshybench import texalign
    ref = texalign.prepare_reference("reference.glb")               # once per reference
    r = texalign.score("generated.glb", ref)
    # {"pattern", "style", "color", "overall", "transform", "registration", ...}

    shared = texalign.prepare_shared_input("input.glb", ref)         # identical-mesh experiment
    r = texalign.score("painted_input.glb", ref, shared)

    texalign.geotex_self("generated.glb")                            # Mesh-Texture Agreement
"""
from .geotex import geotex_self
from .metric import (Reference, SharedInput, fit_to_input, prepare_reference, prepare_shared_input,
                     score)

__all__ = ["Reference", "SharedInput", "fit_to_input", "prepare_reference", "prepare_shared_input",
           "score", "geotex_self"]
