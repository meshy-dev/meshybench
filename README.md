# meshybench

Three benchmarks for image-to-3D generation, each a self-contained module with the exact
implementation behind the published numbers.

| module | measures | needs |
|---|---|---|
| `meshybench.meshalign` | geometry alignment of a generated mesh against its reference mesh | numpy, scipy, trimesh |
| `meshybench.texalign` | texture alignment of a textured model against its reference, and the reference-free Mesh-Texture Agreement | plus torch, nvdiffrast, transformers, Pillow, scikit-image, a CUDA device |
| `meshybench.mesh_details` | detail richness and detail quality of a mesh, reference free | torch, nvdiffrast, a CUDA device |

## Install

```bash
pip install -e ".[test]"
pip install git+https://github.com/NVlabs/nvdiffrast   # not on PyPI
pytest -q                                                # GPU tests skip visibly without CUDA
```

The texture benchmark downloads `facebook/dinov2-base` at a pinned revision on first use.
Draco-compressed GLB files must be decompressed before scoring: trimesh decodes them to empty
geometry, and `meshalign.load_mesh` refuses such a file.

## Geometry Alignment

```python
from meshybench import meshalign
gt = meshalign.prepare_gt("reference.glb")        # once per reference
r = meshalign.score("generated.glb", gt)
# {"proportion": 0.93, "spatial": 0.83, "semantic": 0.76, "overall": 0.84, "signals": {...}}
```

Each mesh is sampled with 400,000 area-uniform points, centred on the sample's centre of mass and
scaled by its root-mean-square radius. The 24 proper axis-aligned rotations are ranked by symmetric
nearest-neighbour distance on a 1,500-point subset, the best four are refined by 30 rigid ICP
iterations on a 20,000-point subset, and the lowest residual is kept. Scale is applied once; ICP
never rescales.

- **Overall Proportion**: intersection over union of the 16^3 occupancy grids of the two samples.
- **Spatial Distribution**: 1 minus the mean 1D Wasserstein distance over 128 fixed unit
  directions between the projected occupied 64^3 voxel centres, divided by 0.18 and clipped to
  [0, 1]. Each 1D distance is read from 512 evenly spaced quantiles of the two projections.
- **Surface Details**: 1 minus the difference between the two-way nearest-neighbour F-scores at
  distance thresholds 0.10 and 0.02 of the RMS radius, on 6,000 query points per side, clipped
  to [0, 1].
- **overall**: the unweighted mean of the three.

A mesh scored against itself gives 1.0 in every dimension.

## Texture Alignment

```python
from meshybench import texalign
ref = texalign.prepare_reference("reference.glb")          # once per reference
r = texalign.score("generated.glb", ref)
# {"color": ..., "style": ..., "pattern": ..., "overall": ..., "transform": ..., ...}

shared = texalign.prepare_shared_input("input.glb", ref)     # every painter textured the same mesh
r = texalign.score("painted_input.glb", ref, shared)

texalign.geotex_self("generated.glb")                        # Mesh-Texture Agreement, reference free
```

The candidate is placed in the reference frame by the geometry alignment registration above and
rendered from six axis-aligned orthographic views at 1554 pixels (unshaded base colour, fixed
half-width of 2 RMS radii). Each view is cut into nine 518-pixel tiles embedded by DINOv2-base,
giving a 111 by 111 patch-token grid per view.

- **Color**: 40,000 area-uniform base-colour samples per side, read per geometry with glTF
  REPEAT wrap. The reference is split into 12 k-means regions of sample positions; per channel
  (hue, saturation, value in CIELCh) a region-mean term and a whole-surface spread term map
  differences through exp(-d / s) with s = 20 for hue and 12 otherwise, combined 0.7 and 0.3,
  and the channels weighted 0.5, 0.3, 0.2.
- **Texture Style** (`style`): pooled foreground tokens; d is the mean distance plus the
  normalised covariance distance; score exp(-max(d - 0.60, 0) / 0.70).
- **Semantic Details** (`pattern`): for each foreground reference patch the best cosine among
  the candidate's patches within 3 cells, averaged over the union foreground per view and over
  the six views; a patch only one side covers scores 0.
- **overall**: the unweighted mean of the three.
- **Mesh-Texture Agreement** (`geotex_self`): on 150,000 surface samples scaled to a unit
  bounding sphere, geometry ridges (curvature above 0.12 over 40 neighbours) and texture edges
  (Lab distance above 8 after two neighbourhood averages) within 0.08 of each other form the
  co-change domain; each member's offset d scores exp(-max(d - 0.012, 0) / 0.035), weighted by
  ridge curvature and edge strength. A uniform albedo or an empty domain raises ValueError.

For an identical-mesh experiment the input mesh is registered once and each painter's file is
mapped onto it by the axis-aligned similarity that reproduces the input vertices, so every
painter is scored on the same registration.

## Mesh Details

```python
from meshybench import mesh_details
res = mesh_details.evaluate("generated.glb")
res.score                      # richness at 1024, the number of record
res.values                     # richness_1024, richness_2048, richness_4096, quality_4096
```

The mesh is rendered as face normals from eight fixed directions into an 8192 by 8192
orthographic buffer with its longest extent spanning 98 percent of the frame.

- **Richness A(N)**: at screen resolution N each pixel takes a stratified 2 by 2 sub-sample of
  its raster block; the pixel is active when the circular spread of the four normals exceeds 2
  degrees. A(N) is the active fraction over fully covered pixels pooled over the eight views.
- **Quality DQ**: at base 512 each block takes a stratified 8 by 8 sub-sample; blocks whose
  spread lies in [2, 20) degrees are detail blocks. Up to 1,500 per view are scored by 1 minus
  the entropy of the perpendicular tilt histogram (0.5 degree bins) relative to the entropy of
  the widest spread the block could show; block scores are averaged inside three spread strata,
  across the strata holding at least 30 blocks, and across views.

Score uncompressed geometry: transport quantisation inflates richness on dense meshes while
staying invisible. A mesh with fewer than 50 active pixels at 1024 gets `status == "too_small"`
and no score.

## Reproducibility

Every stage uses fixed seeds, fixed views and a pinned descriptor revision. Results are
numerically identical across runs on one machine and agree to float32 precision, about 1e-4 on
values in [0, 1], across machines and CUDA devices.
