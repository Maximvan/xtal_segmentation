"""02 - Annotation-free segmentation (GPU filters) of crystal, cells, nuclei, vacuoles, granules.

usage: CUDA_VISIBLE_DEVICES=2 python 02_segment_classical.py run0910|run0110
Normalised grey values (01): crystal ~130, nucleus ~160, cytoplasm ~190, resin ~198 (smooth).
Outputs in 02_Processed/<run>/: ok, crystal, tissue, nuclei, cells, vac_other, granules (.npy)
"""
import sys, time
import numpy as np, tifffile, cc3d, edt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from xtal_lib import (gpu_filter, save, load, proc_dir, gauss, local_std, dilate, erode, eopen, eclose, fill_holes3d,
                      fill2d, keep_large, largest, VX, NT)

run = sys.argv[1]
D = proc_dir(run)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

v = tifffile.imread(D / "img20.tif")
valid = load(D / "valid")
log("loaded", v.shape)
g = gauss(v, 1.5)
sd = local_std(v, 7)
log("gpu filters")

# ---- sample surface: drop coating + fiducial blocks above it
top = int(v.shape[1] * 0.25)
dark = (g[:, :top, :] < 100) & valid[:, :top, :]
has = dark.any(1)
last = top - 1 - np.argmax(dark[:, ::-1, :], axis=1)
surf = np.where(has, last + 3, 0)
surf = ndi.maximum_filter(ndi.median_filter(surf, size=(5, 31)), size=(1, 9))
rows = np.arange(v.shape[1])[None, :, None]
ok = valid & (rows > surf[:, None, :])
ok = erode(ok, 2)
save(D / "ok", ok)
del dark, rows

# ---- crystal: from the U-Net trained on Jasmijn's annotations (14) if available, else the threshold rule
NETCR = (D / "crystal_net.npz").exists()
if NETCR:
    cr = load(D / "crystal_net").astype(bool) & ok
    log("crystal from U-Net (14)")
cr0 = (g < 148) & ok
if not NETCR:
    cr = eopen(cr0, 2)
    cr = largest(cr)
    cr = eclose(cr, 3)
    cr = fill_holes3d(cr)
    # resin-filled holes / pits / cracks inside the crystal belong to the crystal body:
    # close with a 400 nm sphere and fill, but only add voxels that are not textured cytoplasm
    crc = fill_holes3d(eclose(cr, 20))
    cr |= crc & (sd < 4.5)
    cr = fill2d(fill_holes3d(cr))
del cr0
log("crystal", round(cr.sum() * VX**3, 1), "um3")

# ---- tissue (not resin)
# cytoplasm = membrane texture or clearly darker than resin; ignore the crystal edge (its step
# in contrast looks like texture) and smooth near-resin grey (e.g. resin halo next to the crystal)
edge = dilate(cr, 6) & ~cr  # smoothing + 7-px texture window reach ~6 px past the crystal edge
t = (((sd > 4.5) & (g < 194)) | (g < 182)) & ok & ~edge & ~cr
t = eclose(t, 3)
t = eopen(t, 1)
t |= dilate(eopen(t, 3), 7) & edge  # only real (thick) tissue may reach the crystal surface
t = fill2d(t)
t = eopen(t, 2)
t = keep_large(t, 20000)
t |= cr
filled3 = fill_holes3d(t)
log("tissue")

# keep only tissue connected to the crystal (through cells), restrict work to its bounding box
lab = cc3d.connected_components(filled3, connectivity=6)
ids = np.unique(lab[dilate(cr, 2) & filled3])
near = np.isin(lab, ids[ids > 0])
zz, yy, xx = [np.where(near.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
bb = np.s_[max(zz[0] - 5, 0):zz[-1] + 6, max(yy[0] - 5, 0):yy[-1] + 6, max(xx[0] - 5, 0):xx[-1] + 6]
save(D / "bbox", np.array([[s.start, s.stop] for s in bb]))
log("crystal-connected tissue bbox", [(s.start, s.stop) for s in bb])

cells_mask = near & ~cr
save(D / "tissue", t)

# ---- vacuole lumen pockets: resin-like (bright, smooth) holes enclosed by tissue
pockets = (g > 191) & (sd < 4.2) & ok & erode(filled3, 1) & ~dilate(cr, 1)
pockets = eopen(pockets, 1)
save(D / "pockets", pockets)

# ---- nuclei (texture based): chromatin is smooth, cytoplasm is full of membranes/vesicles.
# Cores = smooth (low local std) mid-grey regions; boundary = watershed on the edge map between the core
# and cytoplasm seeds (bright vesicle lumens / resin / crystal), so it stops at the nuclear envelope.
# Light (loose) chromatin patches are added when enclosed by the nucleus in any XY/XZ/YZ section.
from cupyx.scipy import ndimage as cndi
import cupy as cp


def _sdf(a):
    a = cndi.gaussian_filter(a.astype("float32"), 1)
    m = cndi.gaussian_filter(a, 2)
    return cp.sqrt(cp.maximum(cndi.gaussian_filter(a * a, 2) - m * m, 0))


sd1 = gpu_filter(v, _sdf, halo=12)
tex = gauss(sd1, 3)
med = gpu_filter(v, lambda a: cndi.median_filter(a, size=5), halo=4, chunk=64, out_dtype=np.uint8)
br = gpu_filter(v, lambda a: cndi.maximum_filter(cndi.gaussian_filter(a.astype("float32"), 1), size=(1, 7, 7)), halo=4)
core = (tex < 2.5) & (med > 145) & (med < 178) & cells_mask & ~dilate(cr, 5)
core = keep_large(eopen(core, 4), 40000)
elev = gauss(sd1, 0.7)
mk = np.zeros(v.shape, np.uint8)
mk[erode(core, 1)] = 1
mk[~dilate(core, 14) | ~cells_mask | (((br > 184) | (med > 182)) & ~dilate(core, 3))] = 2
del tex, br
zz_, yy_, xx_ = np.where(dilate(core, 15))
bbn = np.s_[zz_.min():zz_.max() + 1, yy_.min():yy_.max() + 1, xx_.min():xx_.max() + 1]
nu = np.zeros(v.shape, bool)
nu[bbn] = watershed(elev[bbn], markers=mk[bbn]) == 1
del elev, mk, sd1
nu = keep_large(eopen(fill_holes3d(nu), 2), 125000)  # > 1 um3
# light-chromatin patches: holes of the (slightly closed) nucleus in XY sections and in 3D. Only
# chromatin-like voxels are added (median grey < 180, i.e. not bright vesicles / cytoplasm), and no XZ/YZ
# section fills (those drew straight, box-like edges between lobes).
nc = eclose(nu, 3)
f = nc.copy()
for i in range(f.shape[0]):
    if f[i].any():
        f[i] = ndi.binary_fill_holes(f[i])
f = fill_holes3d(f)
nu |= f & ~nc & cells_mask & (med < 180)
nu |= nc & cells_mask & (med < 180)  # small gaps closed only where they look like chromatin
nu = keep_large(eopen(fill_holes3d(nu), 1), 125000)
del nc, f
log("nuclei mask", round(nu.sum() * VX**3, 1), "um3")

# group lobes belonging to one cell: lobes within ~200 nm through cytoplasm merge
grp = cc3d.connected_components(dilate(nu, 10) & cells_mask | nu, connectivity=6)
nuc = np.where(nu, grp, 0)
_, inv = np.unique(nuc, return_inverse=True)
nuc = inv.reshape(nu.shape).astype(np.int32)
log("nuclei (grouped)", nuc.max(), [round(c * VX**3, 1) for c in cc3d.statistics(nuc)["voxel_counts"][1:]])
save(D / "nuclei", nuc)

# ---- cell instances: nucleus-seeded watershed on dark membranes, inside the bbox
# elevation = distance to the nearest nucleus (vox) + weak membrane term: prevents one cell flooding
# around the crystal through the thin cytoplasm layer where cell-cell borders are invisible
vb = v[bb].astype(np.float32)
dn = edt.edt(nuc[bb] == 0, parallel=NT)
elev = dn + 0.05 * ndi.gaussian_filter(255 - vb, 2)
del dn
cell_b = watershed(elev, markers=nuc[bb], mask=(cells_mask | pockets & near)[bb])
cell = np.zeros(v.shape, np.int32)
cell[bb] = cell_b
del elev, vb, cell_b
shell = dilate(cr, 3) & ~cr
ids, c = np.unique(cell[shell], return_counts=True)
sel = (ids > 0) & (c > 500)
touch = ids[sel]
log("cells", cell.max(), "touching crystal:", touch.tolist(), c[sel].tolist())
cell = np.where(np.isin(cell, touch), cell, 0)
# relabel 1..n
_, inv = np.unique(cell, return_inverse=True)
cell = inv.reshape(cell.shape).astype(np.int32)
nuc = np.where(cell > 0, nuc, 0)
_, inv = np.unique(np.where(nuc > 0, cell, 0), return_inverse=True)
save(D / "cells", cell)
save(D / "nuclei", np.where(nuc > 0, cell, 0).astype(np.int32))  # nucleus id = its cell id
save(D / "crystal", cr)

# ---- other vacuoles: enclosed pockets inside cells, not touching the crystal shell
pl = cc3d.connected_components(pockets & (cell > 0), connectivity=6)
touching = np.unique(pl[dilate(cr, 8)])
vo = (pl > 0) & ~np.isin(pl, touching) & (nuc == 0)
vo = keep_large(vo, 300)
save(D / "vac_other", vo)

# ---- dense granules: small, compact, darker-than-cytoplasm bodies in the cytoplasm
cyto = (cell > 0) & (nuc == 0) & ~dilate(cr, 3)
gr = (g < 168) & cyto
gr = eopen(gr, 2)
gl = cc3d.connected_components(gr, connectivity=6)
st = cc3d.statistics(gl)
cnt = st["voxel_counts"]
bbx = st["bounding_boxes"]
ext = np.array([[b.stop - b.start for b in bb_] if bb_ is not None else [1, 1, 1] for bb_ in bbx], float)
fill = cnt / np.prod(ext, 1)
elong = ext.max(1) / np.maximum(ext.min(1), 1)
ok_ = (cnt >= 60) & (cnt <= 30000) & (fill > 0.35) & (elong < 3)  # ~100-750 nm, compact
ok_[0] = False
keep = np.where(ok_)[0]
gran = np.isin(gl, keep)
save(D / "granules", gran)
log("granules", len(keep), "other vacuoles", int(vo.sum()))
log("done")
