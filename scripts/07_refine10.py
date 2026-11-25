"""07 - Redraw all boundaries at 10 nm, crystal vacuole, crystal-surface statistics (replaces 04).

The 20 nm segmentation (02) only gives the topology. At 10 nm (bin2) every boundary between
resin / crystal / other tissue / cell i / nucleus i is redrawn by a seeded watershed on the 10 nm
edge map (local std) inside a +-40 nm band around the upsampled 20 nm boundary, so it snaps to the
membrane / nuclear envelope / crystal edge.
Crystal vacuole: per crystal-surface voxel the outward grey profile (shell means, averaged over
neighbouring surface voxels) is scanned for the first step from vacuole content (~165) to cytoplasm
grey (~177); vacuole = crystal surface .. that edge + 20 nm membrane, only where a cell covers the crystal.
Mitochondria from MitoNet 10 nm probabilities; dense granules detected at 10 nm.
usage: CUDA_VISIBLE_DEVICES=5 python 07_refine10.py run0910|run0110
Outputs 02_Processed/<run>/: lab10.npz (uint8 labels, crop), vac10.npz, mito10.npz (uint16 ids), gran10.npz, crop10.json,
        crystal10_surface_stats.json ; 03_Output/<run>/crystal_surface.csv
Labels: 1 resin, 2 crystal, 3 other tissue, 10+i cell i (cytoplasm), 30+i nucleus of cell i.
"""
import sys, json, time
import numpy as np, tifffile, cc3d, edt
import cupy as cp
from cupyx.scipy import ndimage as cndi
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from xtal_lib import save, load, RUNS, proc_dir, out_dir, NT, eopen

run = sys.argv[1]
D, O = proc_dir(run), out_dir(run)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

GREY_STEP = 176.0  # vacuole content ~160 vs cytoplasm ~185; tuned on a hand-traced vacuole outline (run0110, IoU ~0.7)
GREY_DARK = 150.0  # darker than vacuole content (p10 ~152) = granule / dense material against the crystal: stop
DMAX, DMIN, SIG = 30, 2, 4.0  # vacuole search depth 300 nm, min 20 nm; surface smoothing 40 nm
SMOOTH = 1.5  # 3D label smoothing after snapping (removes slice-to-slice noise strips)
BAND = 4  # +-40 nm around the 20 nm boundary

# ---------------- 20 nm labels
cr20 = load(D / "crystal").astype(bool)
cells20 = load(D / "cells").astype(np.int32)
nuc20 = load(D / "nuclei").astype(np.int32)
tis20 = load(D / "tissue").astype(bool)
assert cells20.max() < 20
L20 = np.ones(cr20.shape, np.uint8)
L20[tis20] = 3
L20[cells20 > 0] = 10 + cells20[cells20 > 0]
L20[nuc20 > 0] = 30 + nuc20[nuc20 > 0]
L20[cr20] = 2

# mitochondria (20 nm, MitoNet): prob > 0.5 inside cells, not nucleus/crystal, > 0.005 um3
mp = load(D / "mito_prob20")
mp = np.pad(mp, [(0, max(0, a - b)) for a, b in zip(cr20.shape, mp.shape)])[:cr20.shape[0], :cr20.shape[1], :cr20.shape[2]]
mi = (mp > 128) & (cells20 > 0) & (nuc20 == 0) & ~cr20
mi = ndi.binary_opening(mi, iterations=1)
ml = cc3d.connected_components(mi, connectivity=6)
cnt = cc3d.statistics(ml)["voxel_counts"]
keep = np.where(cnt > 625)[0]; keep = keep[keep > 0]
ml20 = cc3d.connected_components(np.isin(ml, keep), connectivity=6).astype(np.uint16)
save(D / "mito", ml20)
del mp, mi, ml
log("mitochondria", int(ml20.max()))

# ---------------- 10 nm crop = cells + crystal bbox (+80 nm)
raw = tifffile.memmap(RUNS[run]["bin2"])
norm = json.load(open(D / "norm.json"))
resin, gain = np.array(norm["resin"]), norm["gain"]
fg = (cells20 > 0) | cr20
idx = [np.where(fg.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
lo20 = [int(max(i[0] - 4, 0)) for i in idx]
hi20 = [int(min(i[-1] + 5, s, rs // 2)) for i, s, rs in zip(idx, cr20.shape, raw.shape)]
lo10, n10 = [2 * l for l in lo20], [2 * (h - l) for l, h in zip(lo20, hi20)]
json.dump(dict(lo10=lo10, shape10=n10, lo20=lo20, hi20=hi20), open(D / "crop10.json", "w"))
log("crop10", lo10, n10)
import os
DEN = load(D / "den10") if os.environ.get("DEN") == "1" and (D / "den10.npz").exists() else None
log("input", "denoised (08)" if DEN is not None else "raw")
L10 = np.zeros(n10, np.uint8)
vac10 = np.zeros(n10, bool)
gran10 = np.zeros(n10, bool)
dense10 = np.zeros(n10, bool)
del fg

n_surf = n_cov = lum_vox = cr_vox = dense_vox = n_dense = n_vac = 0
cov, thick = {}, []
CH, HALO = 128, 24
for c0 in range(0, n10[0], CH):
    c1 = min(c0 + CH, n10[0])
    a0, a1 = max(0, c0 - HALO) // 2 * 2, min(n10[0], c1 + HALO + 1) // 2 * 2
    sl20 = (slice(lo20[0] + a0 // 2, lo20[0] + a1 // 2), slice(lo20[1], hi20[1]), slice(lo20[2], hi20[2]))
    L = L20[sl20].repeat(2, 0).repeat(2, 1).repeat(2, 2)
    zr = np.arange(lo10[0] + a0, lo10[0] + a1)
    if DEN is not None:
        sub = DEN[a0:a1].astype(np.float32)
    else:
        sub = raw[zr[0]:zr[-1] + 1, lo10[1]:lo10[1] + n10[1], lo10[2]:lo10[2] + n10[2]].astype(np.float32)
        sub = np.clip((sub - resin[np.minimum(zr // 2, len(resin) - 1)][:, None, None]) * gain + 200, 0, 255)

    # edge map (GPU)
    s = cp.asarray(sub)
    g1 = cndi.gaussian_filter(s, (2, 1, 1))  # extra smoothing along Z (milling) against per-slice noise
    m = cndi.gaussian_filter(g1, 1.5)
    E = cp.sqrt(cp.maximum(cndi.gaussian_filter(g1 * g1, 1.5) - m * m, 0))
    Lc = cp.asarray(L)
    bnd = cp.asnumpy(cndi.maximum_filter(Lc, 3) != cndi.minimum_filter(Lc, 3))
    g1, E = cp.asnumpy(g1), cp.asnumpy(E)
    del s, m, Lc
    cp.get_default_memory_pool().free_all_blocks()

    # 1. boundaries snapped to the 10 nm edges
    band = edt.edt(~bnd, parallel=NT) <= BAND
    zz, yy, xx = np.where(band)
    bb = np.s_[zz.min():zz.max() + 1, yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    mk = np.where(band, 0, L)[bb]
    ws = watershed(E[bb], markers=mk, mask=(edt.edt(~band[bb], parallel=NT) <= 1))
    Lb = L[bb]
    Lb[band[bb]] = ws[band[bb]]
    L[bb] = Lb
    del ws, mk, Lb, band, bnd, E
    # soft majority vote: each voxel takes the label with the highest Gaussian-weighted share
    Lc = cp.asarray(L)
    best = cp.zeros(L.shape, cp.float32)
    Ls = cp.zeros(L.shape, cp.uint8)
    for k in np.unique(L):
        f = cndi.gaussian_filter((Lc == int(k)).astype(cp.float32), SMOOTH)
        upd = f > best
        best[upd] = f[upd]; Ls[upd] = int(k)
        del f, upd
    L = cp.asnumpy(Ls)
    del Lc, best, Ls
    cp.get_default_memory_pool().free_all_blocks()
    nucv = L >= 30
    if nucv.any():
        L[nucv & (edt.edt(L >= 10, parallel=NT) <= 2)] -= 20  # nucleus i -> cytoplasm i near the cell edge
    del nucv
    cr = L == 2
    # crystal: keep the main body, fill holes per slice (resin pits inside the crystal)
    lab = cc3d.connected_components(cr, connectivity=6)
    cs = cc3d.statistics(lab)["voxel_counts"]; cs[0] = 0
    if cs.max() > 0:
        crm = ndi.binary_fill_holes(lab == cs.argmax())
        L[cr & ~crm] = 3
        L[crm] = 2
        cr = crm
    del lab
    cell = L >= 10
    cellid = np.where(L >= 30, L - 30, np.where(L >= 10, L - 10, 0)).astype(np.uint8)

    # 2. crystal vacuole (radial first grey step)
    vac = np.zeros(L.shape, bool)
    dense = np.zeros(L.shape, bool)
    k0, k1 = c0 - a0, c0 - a0 + (c1 - c0)
    if cr.any():
        zz, yy, xx = np.where(edt.edt(~cr, parallel=NT) <= DMAX + 2)
        cb = np.s_[zz.min():zz.max() + 1, yy.min():yy.max() + 1, xx.min():xx.max() + 1]
        crb, cellb, gb = cr[cb], cell[cb], g1[cb]
        dcr, ind = ndi.distance_transform_edt(~crb, return_indices=True)
        surf = crb & ~ndi.binary_erosion(crb, np.ones((3, 3, 3)), border_value=1)
        zone = (dcr > 0) & (dcr <= DMAX) & cellb
        fi = np.ravel_multi_index((ind[0][zone], ind[1][zone], ind[2][zone]), crb.shape)
        d = np.rint(dcr[zone]).astype(np.int64)
        gv = gv_ = gb[zone]
        sflat = np.flatnonzero(surf)
        w = cndi.gaussian_filter(cp.asarray(surf, dtype=cp.float32), SIG)
        prof = np.full((sflat.size, DMAX), 255, np.float32)
        for k in range(1, DMAX + 1):
            mk_ = d == k
            Sx = np.bincount(fi[mk_], weights=gv[mk_], minlength=crb.size).reshape(crb.shape).astype(np.float32)
            Cx = np.bincount(fi[mk_], minlength=crb.size).reshape(crb.shape).astype(np.float32)
            Ss = cndi.gaussian_filter(cp.asarray(Sx), SIG).ravel()[cp.asarray(sflat)]
            Cs = cndi.gaussian_filter(cp.asarray(Cx), SIG).ravel()[cp.asarray(sflat)]
            prof[:, k - 1] = cp.asnumpy(cp.where(Cs > 0.05, Ss / cp.maximum(Cs, 1e-6), 255))
            del Sx, Cx, Ss, Cs
        hit = (prof[:, DMIN:] > GREY_STEP) | (prof[:, DMIN:] < GREY_DARK)
        b = np.where(hit.any(1), hit.argmax(1) + DMIN, DMAX)
        # a vacuole needs a real band: >= 60 nm wide and >= 4 grey darker than the 50 nm beyond it;
        # otherwise the crystal is in direct contact (only the 20 nm membrane layer is drawn)
        cs_ = np.concatenate([np.zeros((prof.shape[0], 1), np.float32), np.cumsum(np.minimum(prof, 255), 1)], 1)
        rows = np.arange(b.size)
        inner = (cs_[rows, b] - cs_[rows, DMIN]) / np.maximum(b - DMIN, 1)
        e_ = np.minimum(b + 5, DMAX)
        outer = (cs_[rows, e_] - cs_[rows, b]) / np.maximum(e_ - b, 1)
        b = np.where((b >= 6) & (outer - inner > 4), b, 0).astype(np.float32)  # >= 60 nm: thinner = crystal edge blur
        B = np.zeros(crb.size, np.float32); B[sflat] = b
        B = cp.asarray(B.reshape(crb.shape))
        w2 = cndi.gaussian_filter(cp.asarray(surf, dtype=cp.float32), SIG / 2) + 1e-6
        B2 = cp.asnumpy(cndi.gaussian_filter(B, SIG / 2) / w2).ravel()
        del B, w, w2, prof, hit
        cp.get_default_memory_pool().free_all_blocks()
        bvox = np.zeros(crb.shape, np.float32)
        bvox[zone] = B2[fi]
        lum = zone & (dcr <= bvox) & (bvox >= 3)
        body = crb | lum
        vb = lum | ((edt.edt(~body, parallel=NT) <= 2) & cellb)
        vac[cb] = vb
        # dense material against the crystal: darker than vacuole content, beyond the edge blur, touching the surface zone
        g15b = cp.asnumpy(cndi.gaussian_filter(cp.asarray(sub[cb]), (2, 1.5, 1.5)))
        cand = (g15b < GREY_DARK) & cellb & (dcr >= 3) & (dcr <= DMAX) & ~lum
        cand = ndi.binary_opening(cand, iterations=1)
        ll = cc3d.connected_components(cand, connectivity=6)
        ids = np.unique(ll[cand & (dcr <= 5)]); ids = ids[ids > 0]
        csz = np.bincount(ll.ravel())
        ids = ids[csz[ids] > 1000]
        dn = np.isin(ll, ids)
        dense[cb] |= dn
        del g15b, cand, ll
        # 3. stats on the chunk core (surface voxels with a cell within 60 nm)
        core = np.zeros(crb.shape, bool)
        cz0, cz1 = max(k0 - cb[0].start, 0), max(k1 - cb[0].start, 0)
        core[cz0:cz1] = True
        near = edt.edt(~cellb, parallel=NT) <= 6
        sc = surf & core
        n_surf += int(sc.sum()); n_cov += int((sc & near).sum())
        lab_near = ndi.grey_dilation(cellid[cb], size=7)
        for i, n in zip(*np.unique(lab_near[sc & near], return_counts=True)):
            if i > 0:
                cov[int(i)] = cov.get(int(i), 0) + int(n)
        thick.append((B2.reshape(crb.shape)[sc & near] * 10).astype(np.float16))
        lum_vox += int((lum & core).sum())
        dense_vox += int((dn & core).sum())
        n_dense += int((sc & (edt.edt(~dn, parallel=NT) <= 8)).sum())
        n_vac += int((sc & near & (B2.reshape(crb.shape) >= 3)).sum())
        cr_vox += int((crb & core).sum())
        del dn, dcr, ind, zone, fi, d, gv, gv_, bvox, lum, body, vb, near, lab_near, B2, surf, crb, cellb, gb
    # dense granule candidates (10 nm): dark compact bodies in the cytoplasm, away from the crystal edge
    cyto = (L >= 10) & (L < 30)
    if cyto.any():
        g15 = cp.asnumpy(cndi.gaussian_filter(cp.asarray(sub), (2, 1.5, 1.5)))
        gc = (g15 < 168) & cyto & ~(edt.edt(L != 2, parallel=NT) <= 6)
        gran10[c0:c1] = gc[k0:k1]
        del g15, gc
    L10[c0:c1] = L[k0:k1]
    vac10[c0:c1] = vac[k0:k1]
    dense10[c0:c1] = dense[k0:k1]
    log(f"chunk {c0}-{c1}  surf {n_surf} covered {n_cov}")
    del L, vac, cell, cellid, cr, g1, sub

# small 20-nm-only fragments of a cell/nucleus label that lost contact are left as they are
thick = np.concatenate(thick).astype(np.float32) if thick else np.zeros(0, np.float32)
stats = dict(
    crystal_volume_um3=cr_vox * 1e-6,
    crystal_surface_um2=n_surf * 1e-4,
    covered_fraction=n_cov / max(n_surf, 1),
    vacuole_thickness_nm_median=float(np.median(thick[thick >= 30])) if (thick >= 30).any() else 0.0,  # where a vacuole exists
    vacuole_thickness_nm_p90=float(np.percentile(thick[thick >= 30], 90)) if (thick >= 30).any() else 0.0,
    vacuole_lumen_volume_um3=lum_vox * 1e-6,
    vacuole_surface_fraction=n_vac / max(n_surf, 1),
    dense_on_crystal_volume_um3=dense_vox * 1e-6,
    dense_on_crystal_surface_fraction=n_dense / max(n_surf, 1),
)
stats.update({f"surface_fraction_cell_{i}": n / max(n_surf, 1) for i, n in sorted(cov.items())})
log(stats)
json.dump(stats, open(D / "crystal10_surface_stats.json", "w"), indent=1)
with open(O / "crystal_surface.csv", "w") as fh:
    fh.write("metric,value\n")
    for k, v in stats.items():
        fh.write(f"{k},{v}\n")
save(D / "lab10", L10)
save(D / "vac10", vac10)
save(D / "dense10", dense10)

# mitochondria at 10 nm from MitoNet's 10 nm probabilities (prob > 0.5, cytoplasm, > 0.005 um3)
cytoA = (L10 >= 10) & (L10 < 30)
if (D / "mito_prob10.npz").exists():
    pm = load(D / "mito_prob10")[lo10[0]:lo10[0] + n10[0], lo10[1]:lo10[1] + n10[1], lo10[2]:lo10[2] + n10[2]]
    pm = np.pad(pm, [(0, a - b) for a, b in zip(n10, pm.shape)])
    mi = (pm > 128) & cytoA
    del pm
else:  # fallback: 20 nm instances upsampled
    sl20 = tuple(slice(l, h) for l, h in zip(lo20, hi20))
    mi = (ml20[sl20] > 0).repeat(2, 0).repeat(2, 1).repeat(2, 2) & cytoA
mi = eopen(mi, 1)
ml = cc3d.connected_components(mi, connectivity=6)
cnt = cc3d.statistics(ml)["voxel_counts"]
keep = np.where(cnt > 5000)[0]; keep = keep[keep > 0]
m10 = cc3d.connected_components(np.isin(ml, keep), connectivity=6).astype(np.uint16)
del mi, ml
log("mitochondria 10 nm", int(m10.max()))

# dense granules 10 nm: opened candidates, compact, 100-750 nm (as in 02, voxel counts x8)
gr = eopen(gran10 & (m10 == 0) & ~dense10, 2)
gl = cc3d.connected_components(gr, connectivity=6)
st = cc3d.statistics(gl)
cnt = st["voxel_counts"]
ext = np.array([[b.stop - b.start for b in bb_] if bb_ is not None else [1, 1, 1] for bb_ in st["bounding_boxes"]], float)
fill = cnt / np.prod(ext, 1)
elong = ext.max(1) / np.maximum(ext.min(1), 1)
ok_ = (cnt >= 480) & (cnt <= 240000) & (fill > 0.35) & (elong < 3); ok_[0] = False
g10 = np.isin(gl, np.where(ok_)[0])
save(D / "gran10", g10)
log("granules 10 nm", int(ok_.sum()))
del gr, gl, cytoA
save(D / "mito10", m10)
log("done")
