"""10 - (a) smoother meshes from signed-distance maps, (b) flat map of the crystal surface.

(a) Each 10 nm mask -> signed distance (inside +, outside -), light Gaussian, marching cubes at 0. Sub-voxel
    accurate surfaces without the shrinkage/stair-steps of blurred binary masks. Writes a second scene
    (scene_hq.glb, viewer/scene_<run>_hq.glb.txt) for side-by-side comparison; not published automatically.
(b) Crystal surface voxels are mapped to (position along the crystal axis, angle around it). Per surface point:
    covering cell (within 60 nm), vacuole thickness, dense material within 80 nm. Saved as PNG maps.
usage: python 10_mesh_sdf_surfacemap.py run0910
"""
import sys, json, base64
import numpy as np, edt, trimesh
import fast_simplification as fs
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from xtal_lib import load, proc_dir, out_dir, ROOT, VX, NT

run = sys.argv[1]
D, O = proc_dir(run), out_dir(run)
E = O / "explore"; E.mkdir(exist_ok=True)
c10 = json.load(open(D / "crop10.json"))
OFF, V10 = np.array(c10["lo10"]), VX / 2
L = load(D / "lab10")
cr = L == 2
cells = np.where(L >= 30, L - 30, np.where(L >= 10, L - 10, 0)).astype(np.uint8)
vac = load(D / "vac10").astype(bool)
dense = load(D / "dense10").astype(bool) if (D / "dense10.npz").exists() else np.zeros_like(cr)


def sdf_mesh(mask, step=1, target=None, sigma=0.7):
    if mask.sum() < 50:
        return None
    zz, yy, xx = np.where(mask)
    lo = np.maximum([zz.min() - 4, yy.min() - 4, xx.min() - 4], 0)
    hi = [zz.max() + 5, yy.max() + 5, xx.max() + 5]
    m = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    s = edt.edt(m, parallel=NT) - edt.edt(~m, parallel=NT)
    s = ndi.gaussian_filter(s.astype(np.float32), sigma)[::step, ::step, ::step]
    s = np.pad(s, 1, constant_values=-5)
    v, f, _, _ = marching_cubes(s, 0.0)
    v = ((v - 1) * step + lo + OFF)[:, ::-1] * V10
    v[:, 1] *= -1
    if target and len(f) > target:
        v, f = fs.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=1 - target / len(f))
    me = trimesh.Trimesh(v, f, process=True)
    trimesh.smoothing.filter_taubin(me, iterations=3)
    return me


C = dict(crystal=(86, 180, 233), vacuole=(240, 228, 66), nucleus=(0, 114, 178), mito=(213, 94, 0),
         dense=(0, 158, 115), granules=(80, 80, 80))
CELLC = [(230, 159, 0), (200, 200, 200), (170, 150, 120), (150, 170, 190)]
scene = trimesh.Scene()
areas = {}


def add(name, me, rgb):
    if me is None:
        return
    me.visual.face_colors = [*rgb, 255]
    scene.add_geometry(me, node_name=name, geom_name=name)
    areas[name] = round(float(me.area), 2)
    print(name, len(me.faces), "faces, area", areas[name], flush=True)


ids = [int(i) for i in np.unique(cells) if i > 0]
def regularise(oc, osac):
    """3D consistency for the slice-wise native edges (22 refines every slice on its own -> the edge position
    jitters between slices -> ridges / flakes in 3D). Occupancy is averaged over ~3-5 slices in Z (sub-10 nm XY
    edge is kept), crystal islands removed and holes filled in 3D; the sac is the smoothed crystal+sac envelope
    minus the crystal (a closed shell instead of per-slice strips), small loose pieces removed."""
    import cc3d
    c = ndi.gaussian_filter(oc.astype(np.float32), (1.5, 0.6, 0.6))
    m = c > 0.5
    lab, n = cc3d.connected_components(m, return_N=True, connectivity=6)
    if n > 1:
        sz = np.bincount(lab.ravel())[1:]
        keep = np.zeros(n + 1, bool); keep[1:] = sz >= 0.01 * sz.max()
        c[m & ~keep[lab]] = 0
        m = keep[lab]
    del lab
    h = ndi.binary_fill_holes(m) & ~m
    c[h] = 1; del h, m
    env = np.minimum(oc.astype(np.float32) + osac.astype(np.float32), 1)
    env = ndi.gaussian_filter(env, (2.0, 0.8, 0.8))
    s = np.minimum(env, 1 - c); del env
    ms = s > 0.5
    lab, n = cc3d.connected_components(ms, return_N=True, connectivity=26)
    if n > 1:
        sz = np.bincount(lab.ravel())[1:]
        small = np.zeros(n + 1, bool); small[1:] = sz < 20000          # < 0.02 um3 loose pieces
        s[small[lab]] = 0
    print("regularised: crystal %.1f um3, sac %.1f um3 (before %.1f / %.1f)" % (
        (c > 0.5).sum() * 1e-6, (s > 0.5).sum() * 1e-6, oc.astype(np.float32).sum() * 1e-6,
        osac.astype(np.float32).sum() * 1e-6), flush=True)
    return c, s


def occ_mesh(occ, b10, target):
    """marching cubes at 50 % occupancy of native-resolution masks on the 10 nm grid (sub-voxel accurate)"""
    o = np.pad(ndi.gaussian_filter(occ.astype(np.float32), 0.5), 1)  # (regularised already)
    if o.max() < 0.5:
        return None
    v, f, _, _ = marching_cubes(o, 0.5)
    v = ((v - 1) + np.array([b[0] for b in b10]) + OFF)[:, ::-1] * V10
    v[:, 1] *= -1
    if target and len(f) > target:
        v, f = fs.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=1 - target / len(f))
    me = trimesh.Trimesh(v, f, process=True)
    trimesh.smoothing.filter_taubin(me, iterations=3)
    return me


if (D / "native_occ10.npz").exists():  # crystal + sac from the native-resolution U-Net (18)
    nat = np.load(D / "native_occ10.npz")
    oc_, os_ = regularise(nat["crystal"], nat["sac"])
    add("crystal", occ_mesh(oc_, nat["bbox10"], 50000), C["crystal"])
    add("vacuole_crystal", occ_mesh(os_, nat["bbox10"], 60000), C["vacuole"])
    del oc_, os_
    print("crystal + sac meshes from native resolution", flush=True)
else:
    add("crystal", sdf_mesh(cr, target=50000), C["crystal"])
    add("vacuole_crystal", sdf_mesh(vac & ~cr, target=60000, sigma=0.5), C["vacuole"])
bc, bn = min(100000, 300000 // max(len(ids), 1)), min(45000, 135000 // max(len(ids), 1))  # keep file < 16 MB
for k, i in enumerate(ids):
    add(f"cell_{i}", sdf_mesh(cells == i, step=2, target=bc), CELLC[k % len(CELLC)])
    add(f"nucleus_cell{i}", sdf_mesh(L == 30 + i, step=2, target=bn), C["nucleus"])
add("mitochondria", sdf_mesh(load(D / "mito10") > 0, target=50000, sigma=0.5), C["mito"])
add("granules", sdf_mesh(load(D / "gran10").astype(bool), target=40000, sigma=0.5), C["granules"])
# dense material layer disabled (2026-11-26): picks up membranes / curtaining outside run0910, not validated
for ply in sorted((O / "clem").glob("*.ply")) if (O / "clem").exists() else []:  # CLEM LM channels (19)
    me = trimesh.load(ply)
    add(f"LM_{ply.stem}", me, tuple(int(x) for x in me.visual.face_colors[0][:3]))
scene.export(O / "scene_hq.glb")
(ROOT / f"03_Output/viewer/scene_{run}_hq.glb.txt").write_bytes(base64.b64encode((O / "scene_hq.glb").read_bytes()))
json.dump(areas, open(E / "mesh_hq_areas_um2.json", "w"), indent=1)
print("hq scene MB", round((O / "scene_hq.glb").stat().st_size / 1e6, 1), flush=True)

# ---------------- (b) crystal surface map
zz, yy, xx = np.where(edt.edt(~cr, parallel=NT) <= 32)
bb = np.s_[zz.min():zz.max() + 1, yy.min():yy.max() + 1, xx.min():xx.max() + 1]
crb, cellb, vacb, denb = cr[bb], cells[bb], vac[bb] & ~cr[bb], dense[bb]
del zz, yy, xx
surf = crb & ~ndi.binary_erosion(crb, np.ones((3, 3, 3)), border_value=1)
P = np.argwhere(surf).astype(np.float32)
mu = P.mean(0)
_, _, vt = np.linalg.svd(P[::max(1, len(P) // 200000)] - mu, full_matrices=False)
ax, e1, e2 = vt[0], vt[1], vt[2]
t = (P - mu) @ ax
ang = np.arctan2((P - mu) @ e2, (P - mu) @ e1)
# per surface voxel attributes
dcell = edt.edt(cellb == 0, parallel=NT)
near = dcell[surf] <= 6
cid = ndi.grey_dilation(cellb, size=13)[surf] * near
# vacuole thickness: distance from the crystal of the farthest vacuole voxel owned by each surface voxel
dcr, ind = ndi.distance_transform_edt(~crb, return_indices=True)
sid = -np.ones(crb.shape, np.int64); sid[surf] = np.arange(surf.sum())
own = sid[ind[0][vacb], ind[1][vacb], ind[2][vacb]]
thick = np.zeros(surf.sum(), np.float32)
np.maximum.at(thick, own[own >= 0], dcr[vacb][own >= 0] * 10)
den_near = edt.edt(~denb, parallel=NT)[surf] <= 8
del dcr, ind, sid
# rasterise: x = position along axis (10 nm px -> 40 nm bins), y = angle (2 deg bins)
bx = ((t - t.min()) / 4).astype(int)
by = ((ang + np.pi) / (2 * np.pi) * 180).astype(int).clip(0, 179)
Wm = bx.max() + 1
def raster(val, fn="max"):
    img = np.full((180, Wm), np.nan, np.float32)
    if fn == "max":
        np.fmax.at(img, (by, bx), val.astype(np.float32))
    return img
CC = np.array([(0, 0, 0), (230, 159, 0), (86, 180, 233), (0, 158, 115), (204, 121, 167)], np.uint8)
cov = raster(cid)
covimg = np.full((180, Wm, 3), 255, np.uint8)
okp = ~np.isnan(cov)
covimg[okp] = np.where((cov[okp] > 0)[:, None], CC[np.minimum(cov[okp].astype(int), len(CC) - 1)], (120, 120, 120))
th = raster(thick)
thn = np.clip(np.nan_to_num(th) / 200, 0, 1)
# cividis-like (colourblind safe): dark blue -> yellow
cmap = lambda x: np.stack([np.interp(x, [0, .5, 1], [0, 124, 255]), np.interp(x, [0, .5, 1], [32, 123, 233]),
                           np.interp(x, [0, .5, 1], [77, 120, 69])], -1).astype(np.uint8)
thimg = np.where(okp[..., None], cmap(thn), 255).astype(np.uint8)
dn = raster(den_near.astype(np.float32))
dnimg = np.where(okp[..., None], np.where((np.nan_to_num(dn) > 0)[..., None], np.array([0, 158, 115], np.uint8), np.array([200, 200, 200], np.uint8)), 255).astype(np.uint8)
gap = np.full((6, Wm, 3), 255, np.uint8)
big = np.vstack([covimg, gap, thimg, gap, dnimg]).repeat(3, 0).repeat(3, 1)
Image.fromarray(big).save(E / "crystal_surface_map.png")
np.savez_compressed(E / "crystal_surface_map.npz", cover=cov, thickness_nm=th, dense=dn,
                    bin_um_axis=0.04, bin_deg=2.0)
print("surface map: rows = angle around the crystal axis (0-360 deg), columns = position along axis (40 nm bins)")
print("panels: top covering cell (orange cell 1, sky blue cell 2, grey bare), middle vacuole thickness 0-200 nm "
      "(dark blue -> yellow), bottom dense material within 80 nm (green)")
print("done")
