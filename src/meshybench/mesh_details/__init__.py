"""meshybench.mesh_details: reference-free detail richness and detail quality.

    from meshybench import mesh_details
    res = mesh_details.evaluate("gen.glb")     # needs torch + nvdiffrast on CUDA
    res.score                                  # richness at 1024 (the number of record)
    res.values                                 # richness at 1024/2048/4096 and quality_4096 (DQ)

See metric.py for the definitions, quality.py for DQ and render.py for the eight-view raster.
"""
from .metric import (
    LEVELS,
    MIN_ACTIVE,
    QUALITY_KEY,
    SCORE_LEVEL,
    THRESHOLD_DEG,
    DetailResult,
    evaluate,
    level_stats,
)
from .quality import DQ_BASE, DQ_PER_VIEW, dq64_level
from .render import RASTER, VIEWS, raster_view

__all__ = [
    "LEVELS", "MIN_ACTIVE", "QUALITY_KEY", "SCORE_LEVEL",
    "THRESHOLD_DEG", "RASTER", "VIEWS", "DQ_BASE", "DQ_PER_VIEW", "DetailResult", "evaluate",
    "level_stats", "dq64_level", "raster_view",
]
