"""Score the shipped rifle example and compare with the expected values.

    python examples/run_rifle.py            # needs torch, nvdiffrast and a CUDA device

The reference is an Objaverse model; the two generations were made from input.jpg by Meshy 7
(meshy-7.1, standard geometry resolution) and Tripo 3.1. Every value must agree with
expected.json within the stated tolerance, which is float32 agreement on scores in [0, 1].
"""
import json
import os
import sys

from meshybench import mesh_details, meshalign, texalign

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rifle")
exp = json.load(open(os.path.join(HERE, "expected.json")))
tol = exp["tolerance"]
ref_glb = os.path.join(HERE, exp["reference"])
gt = meshalign.prepare_gt(ref_glb)
ref = texalign.prepare_reference(ref_glb)
worst = 0.0
for name, e in exp["generations"].items():
    gen = os.path.join(HERE, e["file"])
    got = {"geometry": meshalign.score(gen, gt), "texture": texalign.score(gen, ref),
           "mesh_texture_agreement": texalign.geotex_self(gen), "details": mesh_details.evaluate(gen).values}
    print(f"{e['system']}: geometry {got['geometry']['overall']:.4f}  texture {got['texture']['overall']:.4f}  "
          f"mesh-texture agreement {got['mesh_texture_agreement']:.4f}  richness 1024 {got['details']['richness_1024']:.4f}")
    for section in ("geometry", "texture", "details"):
        for k, v in e[section].items():
            d = abs(got[section][k] - v); worst = max(worst, d)
            if d > tol:
                print(f"  MISMATCH {section}.{k}: got {got[section][k]:.5f} expected {v:.5f}")
    d = abs(got["mesh_texture_agreement"] - e["mesh_texture_agreement"]); worst = max(worst, d)
    if d > tol:
        print(f"  MISMATCH mesh_texture_agreement: got {got['mesh_texture_agreement']:.5f} expected {e['mesh_texture_agreement']:.5f}")
print(f"largest difference from expected.json: {worst:.6f} (tolerance {tol})")
sys.exit(0 if worst <= tol else 1)
