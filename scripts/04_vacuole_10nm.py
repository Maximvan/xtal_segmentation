"""04 - Vacuole around the crystal at 10 nm isotropic (bin2 data), plus crystal-surface statistics.

Within a crop around the crystal, processed in Z chunks (with halo):
  1. crystal boundary refined at 10 nm (threshold inside a +-80 nm band of the 20 nm mask)
  2. lumen = electron-lucent, smooth voxels inside the cells within 400 nm of the crystal,
     connected to the crystal surface (space between crystal and the enclosing membrane)
  3. vacuole = crystal + lumen + 20 nm membrane layer (only where a cell is present)
  4. per crystal-surface voxel: covered by a cell (which one) or exposed; local gap thickness
usage: CUDA_VISIBLE_DEVICES=2 python 04_vacuole_10nm.py run0910|run0110
Outputs: 02_Processed/<run>/{vac_xtal, crystal_refined}.npy (20 nm bool), vac_gap20.npy (uint8 gap nm/4),
         crystal10_surface_stats.json ; 03_Output/<run>/crystal_surface.csv
"""
import sys, json, time
import numpy as np, tifffile, cc3d, edt
from scipy import ndimage as ndi
from xtal_lib import save, load, RUNS, proc_dir, out_dir, gauss, local_std, fill_holes3d, NT

run = sys.argv[1]
D, O = proc_dir(run), out_dir(run)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

cr20 = load(D / "crystal")
cells20 = load(D / "cells")
pk20 = load(D / "pockets")
norm = json.load(open(D / "norm.json"))
resin, gain = np.array(norm["resin"]), norm["gain"]
raw = tifffile.memmap(RUNS[run]["bin2"])
Z2, H2, W2 = raw.shape

m = 25  # 500 nm margin (20 nm px)
idx = [np.where(cr20.any(axis=a))[0] for a in ((1, 2), (0, 2), (0, 1))]
lo20 = [max(i[0] - m, 0) for i in idx]
hi20 = [min(i[-1] + m + 1, s) for i, s in zip(idx, cr20.shape)]
# 10 nm crop must be even-sized and inside the raw stack
lo10 = [2 * l for l in lo20]
hi10 = [lo + 2 * ((min(2 * h, s) - lo) // 2) for lo, h, s in zip(lo10, hi20, (Z2, H2, W2))]
n20 = [(h - l) // 2 for l, h in zip(lo10, hi10)]
log("crop10", lo10, hi10)

vac20 = np.zeros(cr20.shape, bool)
crr20 = cr20.copy()
gap20 = np.zeros(cr20.shape, np.uint8)
n_surf = n_cov = 0
cov = {}
gaps = []
lum_vox = 0
cr_vox = 0

CH, HALO = 200, 50  # 10 nm slices per chunk / halo
for c0 in range(0, n20[0] * 2, CH):
    c1 = min(c0 + CH, n20[0] * 2)
    a0, a1 = max(0, c0 - HALO) // 2 * 2, min(n20[0] * 2, c1 + HALO + 1) // 2 * 2
    z20 = slice(lo20[0] + a0 // 2, lo20[0] + a1 // 2)
    yx20 = np.s_[lo20[1]:lo20[1] + n20[1], lo20[2]:lo20[2] + n20[2]]

    def up(a20):
        return a20[z20][(slice(None),) + yx20].repeat(2, 0).repeat(2, 1).repeat(2, 2)

    crU = up(cr20)
    if not crU.any():
        continue
    cellU, pkU = up(cells20), up(pk20)
    zr = np.arange(lo10[0] + a0, lo10[0] + a1)
    sub = raw[zr[0]:zr[-1] + 1, lo10[1]:hi10[1], lo10[2]:hi10[2]].astype(np.float32)
    rz = resin[np.minimum(zr // 2, len(resin) - 1)]
    sub = np.clip((sub - rz[:, None, None]) * gain + 200, 0, 255).astype(np.uint8)
    g = gauss(sub, 2.0)  # at 10 nm resin noise alone gives local std ~7, so use smoothed brightness only
    del sub

    # 1. refine crystal
    d_in = edt.edt(crU, parallel=NT)
    d_o = edt.edt(~crU, parallel=NT)
    cr = (d_in > 8) | (((d_in > 0) | (d_o <= 8)) & (g < 150))
    lab = cc3d.connected_components(cr, connectivity=6)
    cnt = cc3d.statistics(lab)["voxel_counts"]; cnt[0] = 0
    cr = fill_holes3d(lab == np.argmax(cnt))
    del lab, d_in, d_o, crU

    # 2. lumen connected to the crystal surface
    d_out = edt.edt(~cr, parallel=NT)
    inside = (cellU > 0) | pkU
    lum = (g > 190) & inside & (d_out > 0) & (d_out <= 40)
    seed = lum & (d_out <= 6)  # crystal edge blur darkens the first ~40 nm
    ll = cc3d.connected_components(lum, connectivity=6)
    ids = np.unique(ll[seed]); ids = ids[ids > 0]
    lum = np.isin(ll, ids)
    del ll, g

    # 3. vacuole
    body = cr | lum
    vac = body | ((edt.edt(~body, parallel=NT) <= 2) & (cellU > 0))
    gap = np.zeros(cr.shape, np.float32)
    d_b = edt.edt(body, parallel=NT)
    gap[lum] = (d_out[lum] + d_b[lum]) * 10  # nm, local crystal-to-membrane width
    del d_b

    # 4. stats on the chunk core
    k0, k1 = c0 - a0, c0 - a0 + (c1 - c0)
    core = np.s_[k0:k1]
    surf = cr & (edt.edt(cr, parallel=NT) <= 1)
    near = edt.edt(cellU == 0, parallel=NT) <= 6
    gmax = ndi.maximum_filter(gap, size=15)  # lumen starts up to ~70 nm from the surface
    s_c, n_c = surf[core], near[core]
    n_surf += int(s_c.sum()); n_cov += int((s_c & n_c).sum())
    # attribute covered surface voxels to the nearest cell id (within 60 nm)
    lab_near = ndi.grey_dilation(cellU, size=7)[core]
    for i, n in zip(*np.unique(lab_near[s_c & n_c], return_counts=True)):
        if i > 0:
            cov[int(i)] = cov.get(int(i), 0) + int(n)
    gaps.append(gmax[core][s_c & n_c].astype(np.float16))
    lum_vox += int(lum[core].sum()); cr_vox += int(cr[core].sum())

    # back to 20 nm (core part only; chunk bounds are even)
    def down(a, fn):
        a = a[core]
        z, y, x = a.shape
        return fn(a.reshape(z // 2, 2, y // 2, 2, x // 2, 2), axis=(1, 3, 5))
    zz = slice(lo20[0] + c0 // 2, lo20[0] + c0 // 2 + (c1 - c0) // 2)
    vac20[(zz,) + yx20] = down(vac.astype(np.float32), np.mean) >= 0.5
    crr20[(zz,) + yx20] = down(cr.astype(np.float32), np.mean) >= 0.5
    gap20[(zz,) + yx20] = np.clip(down(gap, np.max) / 4, 0, 255).astype(np.uint8)
    log(f"chunk {c0}-{c1}  surf {n_surf}  covered {n_cov}")
    del vac, cr, lum, gap, gmax, body, near, surf, cellU, pkU, d_out

gaps = np.concatenate(gaps).astype(np.float32)
wl = gaps[gaps > 0]
stats = dict(
    crystal_volume_um3=cr_vox * 1e-6,
    crystal_surface_um2=n_surf * 1e-4,
    covered_fraction=n_cov / max(n_surf, 1),
    covered_with_lumen_fraction=float((gaps > 0).mean()) if gaps.size else 0.0,
    lumen_gap_nm_median=float(np.median(wl)) if wl.size else 0.0,
    lumen_gap_nm_p90=float(np.percentile(wl, 90)) if wl.size else 0.0,
    lumen_volume_um3=lum_vox * 1e-6,
)
stats.update({f"surface_fraction_cell_{i}": n / max(n_surf, 1) for i, n in sorted(cov.items())})
log(stats)
json.dump(stats, open(D / "crystal10_surface_stats.json", "w"), indent=1)
with open(O / "crystal_surface.csv", "w") as fh:
    fh.write("metric,value\n")
    for k, v in stats.items():
        fh.write(f"{k},{v}\n")
save(D / "vac_xtal", vac20)
save(D / "crystal_refined", crr20)
save(D / "vac_gap20", gap20)
log("done")
