"""18 - Apply the native-resolution crystal/sac U-Net (17) to one run, after its 10 nm pipeline (07/16/12).

Region: envelope = 10 nm crystal + constrained sac (vac10), dilated 80 nm; predicted at native full resolution
(run0910/0110: 5x5x10 nm, cM 6 nm, 6M 8 nm, 6D2P 6 nm). Native masks are restricted to the envelope and the
sac to cells (10 nm labels mapped up), so the biology constraints of 12 still hold.
Outputs: 02_Processed/<run>/native_{crystal,sac}.npz (bool, packed into the native bbox) + native_meta.json,
         native sac thickness stats merged into crystal10_surface_stats.json (keys native_*),
         02_Processed/<run>/native_occ10.npz: fraction of each 10 nm voxel that is crystal / sac (for meshes).
usage: CUDA_VISIBLE_DEVICES=3 python 18_native_apply.py <run>
"""
import sys, json, time
import numpy as np, tifffile, edt
from scipy import ndimage as ndi
from xtal_lib import ROOT, load, save, proc_dir, NT
import importlib
m17 = importlib.import_module("17_native_sac")

run = sys.argv[1]
D = proc_dir(run)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
FULL = dict(m17.FULL)
FULL["smallXtal_6D2P"] = (m17.RAWD / "EM447_CLEM2_SmallXtal_6D2P_0106/EM447_CLEM2_6D2P_010626.tif", 6.0, 6.0)
src, xy, zn = FULL[run]
st = tifffile.memmap(src)
c10 = json.load(open(D / "crop10.json")); lo, n = c10["lo10"], c10["shape10"]
norm = json.load(open(D / "norm.json")); res, gain = np.array(norm["resin"]), norm["gain"]

L = load(D / "lab10")
env10 = (L == 2) | load(D / "vac10").astype(bool)
env10 = edt.edt(~env10, parallel=NT) <= 8
cell10 = L >= 10
zz, yy, xx = np.where(env10)
b10 = [(int(a.min()), int(a.max()) + 1) for a in (zz, yy, xx)]
sz, sxy = 10.0 / zn, 10.0 / xy       # native px per 10 nm px
# native bbox (absolute native coordinates)
nb = [(int((lo[0] + b10[0][0]) * sz), min(int(np.ceil((lo[0] + b10[0][1]) * sz)), st.shape[0])),
      (int((lo[1] + b10[1][0]) * sxy), min(int(np.ceil((lo[1] + b10[1][1]) * sxy)), st.shape[1])),
      (int((lo[2] + b10[2][0]) * sxy), min(int(np.ceil((lo[2] + b10[2][1]) * sxy)), st.shape[2]))]
shp = [b - a for a, b in nb]
log("native bbox", nb, "shape", shp, f"{xy}x{xy}x{zn} nm", "GVox", round(np.prod(shp) / 1e9, 2))
net = m17.load_net()
dz = m17.neigh(zn)
M = 64  # context margin (native px)
cry = np.zeros(shp, bool); sac = np.zeros(shp, bool)
yi = np.clip(((np.arange(nb[1][0], nb[1][1]) + 0.5) / sxy).astype(int) - lo[1], 0, n[1] - 1)
xi = np.clip(((np.arange(nb[2][0], nb[2][1]) + 0.5) / sxy).astype(int) - lo[2], 0, n[2] - 1)
for k, z in enumerate(range(nb[0][0], nb[0][1])):
    z10 = int((z + 0.5) / sz)
    zc = min(max(z10 - lo[0], 0), n[0] - 1)
    e = env10[zc][np.ix_(yi, xi)]
    if not e.any():
        continue
    zs = np.clip([z - dz, z, z + dz], 0, st.shape[0] - 1)
    y0, y1 = max(nb[1][0] - M, 0), min(nb[1][1] + M, st.shape[1])
    x0, x1 = max(nb[2][0] - M, 0), min(nb[2][1] + M, st.shape[2])
    a = np.stack([st[q, y0:y1, x0:x1] for q in zs]).astype(np.float32)
    ridx = np.minimum((np.floor(zs / sz).astype(int)) // 2, len(res) - 1)
    a = np.clip((a - res[ridx][:, None, None]) * gain + 200, 0, 255)
    p = m17._PRED_FULL(net, a).argmax(0)
    p = p[nb[1][0] - y0:nb[1][0] - y0 + shp[1], nb[2][0] - x0:nb[2][0] - x0 + shp[2]]
    c = cell10[zc][np.ix_(yi, xi)]
    cry[k] = (p == 1) & e
    sac[k] = (p == 2) & e & c & ~cry[k]
    if k % 200 == 0:
        log("slice", k, "/", shp[0], int(cry[k].sum()), int(sac[k].sum()))
# per-slice crystal hole fill + small cleanup (3D ops on GVox volumes are too heavy here)
for k in range(shp[0]):
    if cry[k].any():
        cry[k] = ndi.binary_fill_holes(ndi.binary_opening(cry[k], iterations=1))
        sac[k] &= ~cry[k]
np.savez_compressed(D / "native_crystal.npz", bits=np.packbits(cry, axis=None), shape=np.array(shp))
np.savez_compressed(D / "native_sac.npz", bits=np.packbits(sac, axis=None), shape=np.array(shp))
json.dump(dict(bbox_native=nb, voxel_nm=[zn, xy, xy], lo10=lo, src=str(src)), open(D / "native_meta.json", "w"), indent=1)
# sac thickness at native resolution: distance from the crystal (nm) at the sac's outer boundary
dc = edt.edt(~cry, anisotropy=(zn, xy, xy), parallel=NT, black_border=False).astype(np.float32)
outer = sac & ~ndi.binary_erosion(sac | cry, structure=np.ones((1, 3, 3)))
th = dc[outer]
del dc
S = json.load(open(D / "crystal10_surface_stats.json"))
S.update(native_voxel_nm=[zn, xy, xy],
         native_crystal_volume_um3=float(cry.sum() * zn * xy * xy * 1e-9),
         native_sac_volume_um3=float(sac.sum() * zn * xy * xy * 1e-9),
         native_sac_thickness_nm_median=float(np.median(th[th >= 20])) if (th >= 20).any() else 0.0,
         native_sac_thickness_nm_p90=float(np.percentile(th[th >= 20], 90)) if (th >= 20).any() else 0.0)
json.dump(S, open(D / "crystal10_surface_stats.json", "w"), indent=1)
log({k: v for k, v in S.items() if k.startswith("native")})
# occupancy on the 10 nm grid inside the 10 nm bbox (fraction of native voxels): sub-voxel accurate meshes in 10
bz, by, bx = [b - a for a, b in b10]
zi10 = np.clip(((np.arange(nb[0][0], nb[0][1]) + 0.5) / sz).astype(int) - lo[0] - b10[0][0], 0, bz - 1)
yb = yi - b10[1][0]; xb = xi - b10[2][0]
ys = np.flatnonzero(np.r_[True, np.diff(yb) != 0]); xs = np.flatnonzero(np.r_[True, np.diff(xb) != 0])
uy, ux = yb[ys], xb[xs]
ok_y, ok_x = (uy >= 0) & (uy < by), (ux >= 0) & (ux < bx)
cnt2 = np.add.reduceat(np.add.reduceat(np.ones((shp[1], shp[2]), np.float32), ys, 0), xs, 1)
occ = {}
for name, msk in (("crystal", cry), ("sac", sac)):
    acc = np.zeros((bz, by, bx), np.float32); cz = np.zeros(bz, np.float32)
    for k in range(shp[0]):
        s2 = np.add.reduceat(np.add.reduceat(msk[k].astype(np.float32), ys, 0), xs, 1) / cnt2
        acc[zi10[k]][np.ix_(uy[ok_y], ux[ok_x])] += s2[np.ix_(ok_y, ok_x)]
        cz[zi10[k]] += 1
    occ[name] = (acc / np.maximum(cz, 1)[:, None, None]).astype(np.float16)
    log(name, "occupancy done")
np.savez_compressed(D / "native_occ10.npz", crystal=occ["crystal"], sac=occ["sac"], bbox10=np.array(b10))
log("done")
