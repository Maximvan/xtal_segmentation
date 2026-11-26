"""22 - Refine crystal and sac edges at NATIVE XY resolution in a thin band around the 10 nm result.

Per 10 nm slice (Z stays 10 nm), on the full-resolution image (run0910/0110 5 nm, cM 6 nm, 6M 8 nm XY):
  crystal: inside a +-20 nm band around the 10 nm crystal edge, threshold halfway between this slice's crystal
           grey (eroded interior) and the local surroundings (40-80 nm outside) -> sub-10 nm edge
  sac    : 2D marker watershed on image darkness inside a +-40 nm band around the 10 nm sac's outer edge;
           markers = crystal + sac interior vs. tissue well outside; the line settles on the darkest ridge
           = the enclosing membrane
Output: native_occ10.npz (fraction crystal / sac per 10 nm voxel, bbox10) -> 10 builds meshes from it;
        native_edges.json (summary). Masks of the 10 nm labels are not changed.
usage: python 22_native_edges.py <run> [eval]
"""
import sys, json, time
import numpy as np, tifffile, edt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from joblib import Parallel, delayed
from xtal_lib import ROOT, load, proc_dir, NT

run = sys.argv[1]
D = proc_dir(run)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
RAWD = ROOT / "01_Input/FibSEM xtals"
FULL = {"run0910": (RAWD / "2026_10_01_JasmijnB_FIBSEM_CLEM1/run_0910/EM447_CLEM1_0910.tif", 5.0, 10.0),
        "run0110": (RAWD / "2026_10_01_JasmijnB_FIBSEM_CLEM1/run_0110/EM_447_CLEM1_1001.tif", 5.0, 10.0),
        "largeXtal_cM": (RAWD / "EM447_largeXtal_cM_1205.tif", 6.0, 6.0),
        "smallXtal_6M": (RAWD / "EM447_CLEM2_smallXtal_6M_0408/EM447_CLEM2_SmallX_6m.tif", 8.0, 8.0)}
src, xy, zn = FULL[run]
s = 10.0 / xy                       # native px per 10 nm px (XY)
c10 = json.load(open(D / "crop10.json")); lo, n = c10["lo10"], c10["shape10"]
L = load(D / "lab10")
CR = L == 2
VA = load(D / "vac10").astype(bool) & ~CR
CELL = L >= 10
env = edt.edt(~(CR | VA), parallel=NT) <= 8
zz, yy, xx = np.where(env)
b10 = [(int(a.min()), int(a.max()) + 1) for a in (zz, yy, xx)]
log("bbox10", b10, "native px per 10 nm", s)


def refine_slice(k):
    """k: slice index inside b10 -> (crystal occupancy, sac occupancy) on the 10 nm bbox grid"""
    z = b10[0][0] + k
    st = tifffile.memmap(src)
    zabs = lo[0] + z
    zn_i = int(np.clip(round((zabs + 0.5) * 10.0 / zn - 0.5), 0, st.shape[0] - 1))   # native slice
    y0, y1, x0, x1 = b10[1][0], b10[1][1], b10[2][0], b10[2][1]
    cr10, va10, ce10 = CR[z, y0:y1, x0:x1], VA[z, y0:y1, x0:x1], CELL[z, y0:y1, x0:x1]
    H10, W10 = cr10.shape
    if not (cr10.any() or va10.any()):
        return np.zeros((H10, W10), np.float16), np.zeros((H10, W10), np.float16)
    Y0, X0 = int((lo[1] + y0) * s), int((lo[2] + x0) * s)
    Hn, Wn = int(round(H10 * s)), int(round(W10 * s))
    a = np.asarray(st[zn_i, Y0:Y0 + Hn, X0:X0 + Wn]).astype(np.float32)
    Hn, Wn = a.shape
    yi = np.minimum(((np.arange(Hn) + 0.5) / s).astype(int), H10 - 1)
    xi = np.minimum(((np.arange(Wn) + 0.5) / s).astype(int), W10 - 1)
    up = lambda m: m[np.ix_(yi, xi)]
    crU, vaU, ceU = up(cr10), up(va10), up(ce10)
    g = ndi.gaussian_filter(a, 1.5 * s / 2)          # ~7.5 nm smoothing
    # --- crystal edge: local half-way threshold in a +-20 nm band
    crN = crU.copy()
    if crU.any():
        din = edt.edt(crU, black_border=False); dout = edt.edt(~crU, black_border=False)
        band = (din <= 2 * s) & crU | (dout <= 2 * s) & ~crU
        inner = din > 4 * s
        ring = (dout > 4 * s) & (dout <= 8 * s) & (ceU | vaU)
        if inner.sum() > 200 and ring.sum() > 200:
            # local outside level: median filter of the ring values spread over the band (coarse grid)
            ci = np.median(g[inner])
            ov = np.where(ring, g, np.nan)
            co = ndi.generic_filter(ov[::8, ::8], np.nanmedian, size=9, mode="nearest")
            co = np.where(np.isnan(co), np.nanmedian(ov), co)
            co = ndi.zoom(co, (Hn / co.shape[0], Wn / co.shape[1]), order=1)[:Hn, :Wn]
            thr = (ci + co) / 2
            crN = (crU & ~band) | (band & (g < thr))
            crN = ndi.binary_opening(crN, iterations=1)
            lab, nl = ndi.label(crN)
            if nl > 1:
                core = np.unique(lab[inner & crN]); core = core[core > 0]
                crN = np.isin(lab, core) if core.size else crN
            crN = ndi.binary_fill_holes(crN)
    # --- sac outer edge: watershed on darkness in a +-40 nm band
    vaN = vaU & ~crN
    if vaU.any():
        body = vaU | crU
        dI = edt.edt(body, black_border=False); dO = edt.edt(~body, black_border=False)
        mk = np.zeros((Hn, Wn), np.int32)
        mk[(crN | (vaU & (dI > 4 * s)))] = 1
        mk[(dO > 4 * s) & ~crN] = 2
        mask = (dI <= 8 * s) & body | (dO <= 8 * s)
        dark = ndi.gaussian_filter(255 - a, 1.0 * s / 2)
        ws = watershed(dark, markers=np.where(mask, mk, 0), mask=mask | (mk > 0))
        vaN = ((ws == 1) | (body & ~mask)) & ~crN & (ceU | vaU)
    # back to 10 nm occupancy
    ys = np.flatnonzero(np.r_[True, np.diff(yi) != 0]); xs = np.flatnonzero(np.r_[True, np.diff(xi) != 0])
    cnt = np.add.reduceat(np.add.reduceat(np.ones((Hn, Wn), np.float32), ys, 0), xs, 1)
    oc = np.zeros((H10, W10), np.float32); ov_ = np.zeros((H10, W10), np.float32)
    oc[np.ix_(yi[ys], xi[xs])] = np.add.reduceat(np.add.reduceat(crN.astype(np.float32), ys, 0), xs, 1) / cnt
    ov_[np.ix_(yi[ys], xi[xs])] = np.add.reduceat(np.add.reduceat(vaN.astype(np.float32), ys, 0), xs, 1) / cnt
    return oc.astype(np.float16), ov_.astype(np.float16)


nz = b10[0][1] - b10[0][0]
res = Parallel(n_jobs=24, batch_size=4, verbose=0)(delayed(refine_slice)(k) for k in range(nz))
occ_c = np.stack([r[0] for r in res]); occ_s = np.stack([r[1] for r in res])
np.savez_compressed(D / "native_occ10.npz", crystal=occ_c, sac=occ_s, bbox10=np.array(b10))
sl = tuple(slice(a, b) for a, b in b10)
summ = dict(native_xy_nm=xy,
            crystal_um3_10nm=float(CR[sl].sum() * 1e-6), crystal_um3_native=float(occ_c.astype(np.float32).sum() * 1e-6),
            sac_um3_10nm=float(VA[sl].sum() * 1e-6), sac_um3_native=float(occ_s.astype(np.float32).sum() * 1e-6))
json.dump(summ, open(D / "native_edges.json", "w"), indent=1)
log(summ)
