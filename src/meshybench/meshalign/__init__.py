"""meshalign: geometry alignment of a generated mesh against its reference.

    from meshybench import meshalign
    gt = meshalign.prepare_gt("reference.glb")          # once per reference
    r = meshalign.score("generated.glb", gt)
    # {"proportion": ..., "spatial": ..., "semantic": ..., "overall": ..., "signals": {...}}
"""
from .loading import load_mesh
from .metric import DIMS, GroundTruth, all_signals, prepare_gt, score
from .registration import norm_params, register
from .sampling import dense_sample

__all__ = ["load_mesh", "score", "prepare_gt", "all_signals", "GroundTruth", "DIMS",
           "register", "norm_params", "dense_sample"]
