"""Shrink a viewer scene below the 16 MB artifact file limit: simplify every mesh by the same factor.
usage: python shrink_glb.py <run> [target_MB=14]"""
import sys, base64
import numpy as np, trimesh, fast_simplification as fs
from xtal_lib import ROOT
run = sys.argv[1]; tgt = float(sys.argv[2]) if len(sys.argv) > 2 else 14.0
glb = ROOT / f"03_Output/{run}/scene_hq.glb"; txt = ROOT / f"03_Output/viewer/scene_{run}_hq.glb.txt"
cur = txt.stat().st_size / 1e6
if cur <= tgt:
    raise SystemExit(f"{run}: {cur:.1f} MB, nothing to do")
sc = trimesh.load(glb, force="scene")
keep = 0.92 * tgt / cur
for name, g in list(sc.geometry.items()):
    if not hasattr(g, "faces") or len(g.faces) < 2000:
        continue
    v, f = fs.simplify(g.vertices.astype(np.float32), g.faces.astype(np.int32), target_reduction=1 - keep)
    m = trimesh.Trimesh(v, f, process=False)
    m.visual.vertex_colors = np.tile(g.visual.main_color, (len(v), 1))
    sc.geometry[name] = m
data = sc.export(file_type="glb")
glb.write_bytes(data); txt.write_text(base64.b64encode(data).decode())
print(run, f"{cur:.1f} -> {txt.stat().st_size / 1e6:.1f} MB")
