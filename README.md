# meshybench

We have three benchmarks for image-to-3D generation. Each benchmark is a self-contained module that
contains the exact implementation behind the published numbers.

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
Draco-compressed GLB files must be decompressed before scoring. Trimesh decodes them to empty
geometry, and `meshalign.load_mesh` refuses such a file.

## Geometry Alignment

```python
from meshybench import meshalign
gt = meshalign.prepare_gt("reference.glb")        # once per reference
r = meshalign.score("generated.glb", gt)
# {"proportion": 0.93, "spatial": 0.83, "semantic": 0.76, "overall": 0.84, "signals": {...}}
```

Each mesh is sampled with 400,000 area-uniform points, centered on the sample's center of mass, and
scaled by its root-mean-square radius. The 24 proper axis-aligned rotations are ranked by symmetric
nearest-neighbor distance on a 1,500-point subset. The pipeline refines the best four rotations
with 30 rigid ICP iterations on a 20,000-point subset and keeps the lowest residual. Scale is
applied once, and ICP never rescales.

- **Overall Proportion**: the intersection over union of the 16^3 occupancy grids of the two
  samples.
- **Spatial Distribution**: calculated as 1 minus the mean 1D Wasserstein distance over 128 fixed
  unit directions between the projected occupied 64^3 voxel centers, divided by 0.18 and clipped to
  [0, 1]. Each 1D distance is read from 512 evenly spaced quantiles of the two projections.
- **Surface Details**: 1 minus the difference between the two-way nearest-neighbor F-scores at
  distance thresholds 0.10 and 0.02 of the RMS radius, on 6,000 query points per side, clipped to
  [0, 1].
- **overall**: the unweighted mean of the three.

A mesh scored against itself returns 1.0 in every dimension.

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

The geometry alignment registration above places the candidate in the reference frame. The system
renders the candidate from six axis-aligned orthographic views at 1554 pixels with unshaded base
color and a fixed half-width of 2 RMS radii. The system cuts each view into nine 518-pixel tiles.
DINOv2-base embeds these tiles, producing a 111 by 111 patch-token grid per view.

- **Color**: The pipeline reads 40,000 area-uniform base-color samples per side per geometry with
  glTF REPEAT wrap. The reference is split into 12 k-means regions of sample positions. For each
  channel of hue, saturation, and value in CIELCh, a region-mean term and a whole-surface spread
  term map differences through `exp(-d / s)` with `s = 20` for hue and `12` otherwise. The
  calculation combines the terms with weights of 0.7 and 0.3, and it weights the channels at 0.5,
  0.3, and 0.2.
- **Texture Style** (`style`): pooled foreground tokens; d is the mean distance plus the normalized
  covariance distance; score exp(-max(d - 0.60, 0) / 0.70).
- **Semantic Details** (`pattern`): the best cosine similarity among candidate patches within 3
  cells for each foreground reference patch. The calculation averages these values over the union
  foreground per view and over the six views. A patch that only one side covers scores 0.
- **overall**: the unweighted mean of the three.
- **Mesh-Texture Agreement** (`geotex_self`): calculated on 150,000 surface samples scaled to a unit
  bounding sphere. Geometry ridges have curvature above 0.12 over 40 neighbors, and texture edges
  have Lab distance above 8 after two neighborhood averages. Geometry ridges and texture edges
  within 0.08 of each other form the co-change domain. Each member's offset `d` scores `exp(-max(d -
  0.012, 0) / 0.035)`, weighted by ridge curvature and edge strength. A uniform albedo or an empty
  domain raises `ValueError`.

We register the input mesh once for an identical-mesh experiment. We map each painter's file onto
this mesh through the axis-aligned similarity that reproduces the input vertices, so every painter
is scored on the same registration.

## Mesh Details

```python
from meshybench import mesh_details
res = mesh_details.evaluate("generated.glb")
res.score                      # richness at 1024, the number of record
res.values                     # richness_1024, richness_2048, richness_4096, quality_4096
```

The mesh is rendered as face normals from eight fixed directions into an 8192 by 8192 orthographic
buffer. Its longest extent spans 98 percent of the frame.

- **Richness A(N)**: the active fraction over fully covered pixels pooled over the eight views. At
  screen resolution `N`, each pixel takes a stratified 2 by 2 sub-sample of its raster block. The
  pixel is active when the circular spread of the four normals exceeds 2 degrees.
- **Quality DQ**: at base 512, each block takes a stratified 8 by 8 sub-sample. Blocks whose spread
  lies in [2, 20) degrees are detail blocks. The metric scores up to 1,500 detail blocks per view by
  1 minus the entropy of the perpendicular tilt histogram (0.5 degree bins) relative to the entropy
  of the widest spread the block could show. The metric averages block scores inside three spread
  strata, across the strata holding at least 30 blocks, and across views.

We score uncompressed geometry. Transport quantization inflates richness on dense meshes while
staying invisible. A mesh with fewer than 50 active pixels at 1024 gets `status == "too_small"` and
receives no score.

## Shipped Example

The directory `examples/rifle/` contains one Objaverse reference model (`reference.glb`, 140,482
faces, one 4K texture), the rendered input view (`input.jpg`), and two generations made from it:
`meshy7.glb` (Meshy 7, meshy-7.1 at standard geometry resolution) and `tripo31.glb` (Tripo 3.1). The
meshes are stored with Git LFS.

```bash
python examples/run_rifle.py
```

The script scores both generations with the three benchmarks and compares every value with
`examples/rifle/expected.json` at the float32 tolerance. The expected values are given in percent:

| benchmark | dimension | Meshy 7 | Tripo 3.1 |
|---|---|---|---|
| geometry alignment | Overall Proportion | 100.0 | 92.7 |
| geometry alignment | Spatial Distribution | 90.0 | 84.0 |
| geometry alignment | Surface Details | 73.4 | 60.7 |
| geometry alignment | overall | 87.8 | 79.1 |
| texture alignment | Color | 70.0 | 68.9 |
| texture alignment | Texture Style | 73.5 | 60.9 |
| texture alignment | Semantic Details | 67.7 | 56.1 |
| texture alignment | overall | 70.4 | 62.0 |
| mesh-texture agreement | | 91.1 | 90.0 |
| mesh details | richness 1024 | 39.4 | 37.7 |
| mesh details | richness 2048 | 26.0 | 23.7 |
| mesh details | richness 4096 | 15.7 | 13.5 |
| mesh details | DQ 4096 | 51.9 | 60.0 |

The object displays the returned metrics and lets a user verify an installation. It does not compare
the two systems.

## Reproducibility

Every stage is configured with fixed seeds, fixed views, and a pinned descriptor revision. Runs on
one machine produce numerically identical results. Across machines and CUDA devices, the results
agree to float32 precision, about 1e-4 on values in [0, 1].
