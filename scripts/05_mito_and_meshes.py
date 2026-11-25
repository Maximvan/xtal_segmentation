"""05 - Mitochondria instances, surface meshes, volumes and viewer data for one run.

Mito: MitoNet prob > 0.5, inside crystal-contacting cells, outside nucleus/crystal, > 0.005 um3.
Meshes: marching cubes on lightly smoothed 20 nm masks (um; X = cut-face X, Y = up, Z = milling),
simplified to a face budget that keeps the viewer file < ~11 MB, one GLB + per-structure PLY.
usage: python 05_mito_and_meshes.py run0910|run0110
Outputs: 02_Processed/<run>/mito.npz ; 03_Output/<run>/{scene.glb, meshes/*.ply, volumes.csv}
         03_Output/viewer/{scene_<run>.glb.txt, data_<run>.json}
"""
import sys, csv, json, base64
import numpy as np, cc3d, trimesh
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import fast_simplification as fs
from xtal_lib import save, load, proc_dir, out_dir, ROOT, VX

run = sys.argv[1]
D, O = proc_dir(run), out_dir(run)
(O / "meshes").mkdir(exist_ok=True)
V = ROOT / "03_Output/viewer"
V.mkdir(parents=True, exist_ok=True)

stats10 = json.load(open(D / "crystal10_surface_stats.json")) if (D / "crystal10_surface_stats.json").exists() else {}
TEN = (D / "lab10.npz").exists()
if TEN:  # 10 nm refined labels from 07 (crop); granules / other vacuoles stay 20 nm
    c10 = json.load(open(D / "crop10.json"))
    L = load(D / "lab10")
    cr = L == 2
    cells = np.where(L >= 30, L - 30, np.where(L >= 10, L - 10, 0)).astype(np.int32)
    nuc = np.where(L >= 30, L - 30, 0).astype(np.int32)
    vx = load(D / "vac10").astype(bool)
    ml = load(D / "mito10").astype(np.int32)
    del L
    OFF, VXM = np.array(c10["lo10"]), VX / 2
else:
    cr = (load(D / "crystal_refined") if (D / "crystal_refined.npz").exists() else load(D / "crystal")).astype(bool)
    cells = load(D / "cells").astype(np.int32)
    nuc = load(D / "nuclei").astype(np.int32)  # voxel value = id of the cell the nucleus belongs to
    vx = load(D / "vac_xtal").astype(bool)
    mp = load(D / "mito_prob20")
    mp = np.pad(mp, [(0, max(0, a - b)) for a, b in zip(cr.shape, mp.shape)])[:cr.shape[0], :cr.shape[1], :cr.shape[2]]
    mi = (mp > 128) & (cells > 0) & (nuc == 0) & ~cr
    mi = ndi.binary_opening(mi, iterations=1)
    ml = cc3d.connected_components(mi, connectivity=6)
    cnt = cc3d.statistics(ml)["voxel_counts"]
    keep = np.where(cnt > 625)[0]
    keep = keep[keep > 0]
    ml = cc3d.connected_components(np.isin(ml, keep), connectivity=6).astype(np.int32)
    save(D / "mito", ml)
    OFF, VXM = np.zeros(3), VX
vo = load(D / "vac_other").astype(bool)
GR10 = TEN and (D / "gran10.npz").exists()
gr = load(D / "gran10" if GR10 else D / "granules").astype(bool)
print("mitochondria", ml.max(), "10nm" if TEN else "20nm", flush=True)

cell_ids = [int(i) for i in np.unique(cells) if i > 0]
nc = len(cell_ids)
BUDGET = 470_000  # total faces
b_cell = int(BUDGET * 0.45 / max(nc, 1))
b_nuc = int(BUDGET * 0.2 / max(nc, 1))


def mesh(mask, step=2, target=None, sigma=1.0, vx=None, off=None):
    vx = VXM if vx is None else vx
    off = OFF if off is None else off
    if mask.sum() < 50:
        return None
    zz, yy, xx = np.where(mask)
    lo = np.maximum([zz.min() - 3, yy.min() - 3, xx.min() - 3], 0)
    hi = [zz.max() + 4, yy.max() + 4, xx.max() + 4]
    sub = ndi.gaussian_filter(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32), sigma)
    sub = np.pad(sub[::step, ::step, ::step], 1)
    if sub.max() < 0.5:
        return None
    v, f, _, _ = marching_cubes(sub, 0.5)
    v = ((v - 1) * step + lo + off)[:, ::-1] * vx
    v[:, 1] *= -1
    if target and len(f) > target:
        v, f = fs.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=1 - target / len(f))
    m = trimesh.Trimesh(v, f, process=True)
    trimesh.smoothing.filter_taubin(m, iterations=5)
    return m


C = dict(dense=(0, 158, 115), crystal=(86, 180, 233), vacuole=(240, 228, 66), nucleus=(0, 114, 178),
         mito=(213, 94, 0), vac_other=(204, 121, 167), granules=(80, 80, 80))
CELLC = [(230, 159, 0), (200, 200, 200), (170, 150, 120), (150, 170, 190), (210, 190, 160),
         (130, 130, 130), (190, 160, 200), (160, 190, 160)]
scene = trimesh.Scene()
vol = {}


def add(name, m, rgb):
    if m is None:
        return
    m.visual.face_colors = [*rgb, 255]
    scene.add_geometry(m, node_name=name, geom_name=name)
    m.export(O / "meshes" / f"{name}.ply")
    print(name, len(m.faces), flush=True)


u3 = VXM ** 3
mc = mesh(cr, target=50000)
add("crystal", mc, C["crystal"]); vol["crystal"] = cr.sum() * u3
if mc is not None and stats10:
    stats10["crystal_surface_um2"] = float(mc.area)  # mesh area; voxel-face counts overestimate
add("vacuole_crystal", mesh(vx | cr, step=1 if TEN else 2, target=60000, sigma=0.8), C["vacuole"]); vol["vacuole_crystal"] = (vx & ~cr).sum() * u3
for k, i in enumerate(cell_ids):
    cm = cells == i
    add(f"cell_{i}", mesh(cm, target=b_cell), CELLC[k % len(CELLC)]); vol[f"cell_{i}"] = cm.sum() * u3
    nm = nuc == i
    add(f"nucleus_cell{i}", mesh(nm, target=b_nuc), C["nucleus"]); vol[f"nucleus_cell{i}"] = nm.sum() * u3
add("mitochondria", mesh(ml > 0, step=1 if not TEN else 2, target=50000, sigma=0.8), C["mito"]); vol["mitochondria"] = (ml > 0).sum() * u3
z0 = np.zeros(3)
if TEN and (D / "dense10.npz").exists():
    dn = load(D / "dense10").astype(bool)
    add("dense_on_crystal", mesh(dn, step=1, target=40000, sigma=0.8), C["dense"]); vol["dense_on_crystal"] = dn.sum() * u3
    del dn
if GR10:
    add("granules", mesh(gr, step=1, target=40000, sigma=0.8), C["granules"]); vol["granules"] = gr.sum() * u3
else:
    add("granules", mesh(gr, step=1, target=40000, sigma=0.8, vx=VX, off=z0), C["granules"]); vol["granules"] = gr.sum() * VX**3
add("other_vacuoles", mesh(vo, step=1, target=40000, sigma=0.8, vx=VX, off=z0), C["vac_other"]); vol["other_vacuoles"] = vo.sum() * VX**3

per_cell = {}
for i in cell_ids:
    cm = cells == i
    per_cell[i] = dict(mito_n=int(len(np.unique(ml[cm & (ml > 0)]))))

scene.export(O / "scene.glb")
with open(O / "volumes.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["structure", "volume_um3"])
    for k, v in vol.items():
        w.writerow([k, round(float(v), 3)])
    for i, d in per_cell.items():
        for k, v in d.items():
            w.writerow([f"cell_{i}_{k}", v])
(V / f"scene_{run}.glb.txt").write_bytes(base64.b64encode((O / "scene.glb").read_bytes()))
json.dump(dict(volumes={k: round(float(v), 3) for k, v in vol.items()}, per_cell=per_cell, surface=stats10),
          open(V / f"data_{run}.json", "w"), indent=1)
print("written", round((O / "scene.glb").stat().st_size / 1e6, 1), "MB")
