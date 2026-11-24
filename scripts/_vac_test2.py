"""scratch: vacuole only where an enclosing membrane ridge follows the grey band; dense material on the crystal."""
import sys, json
import numpy as np, tifffile, cc3d, edt
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.segmentation import find_boundaries
from xtal_lib import RUNS

S = sys.argv[1]
R_THR = float(sys.argv[2])
DENSE = float(sys.argv[3])
DMAX, DMIN, SIG = 25, 2, 4.0


def profiles(crb, vals_list, zone, dcr, ind):
    surf = crb & ~ndi.binary_erosion(crb, np.ones((3, 3, 3)), border_value=1)
    fi = np.ravel_multi_index((ind[0][zone], ind[1][zone], ind[2][zone]), crb.shape)
    d = np.rint(dcr[zone]).astype(np.int64)
    sflat = np.flatnonzero(surf)
    out = [np.full((sflat.size, DMAX), np.nan, np.float32) for _ in vals_list]
    for k in range(1, DMAX + 1):
        m = d == k
        Cs = ndi.gaussian_filter(np.bincount(fi[m], minlength=crb.size).reshape(crb.shape).astype(np.float32), SIG).ravel()[sflat]
        for o, v in zip(out, vals_list):
            Ss = ndi.gaussian_filter(np.bincount(fi[m], weights=v[zone][m], minlength=crb.size).reshape(crb.shape).astype(np.float32), SIG).ravel()[sflat]
            o[:, k - 1] = np.where(Cs > 0.05, Ss / np.maximum(Cs, 1e-6), np.nan)
    return surf, sflat, fi, out


def test(run, z20, ys, xs, name):
    raw = tifffile.memmap(RUNS[run]["bin2"])
    n = json.load(open(f"../02_Processed/{run}/norm.json"))
    z = 2 * z20
    zr = np.arange(z - 20, z + 21)
    sub = raw[zr[0]:zr[-1] + 1, ys, xs].astype(np.float32)
    sub = (sub - np.array(n["resin"])[zr // 2][:, None, None]) * n["gain"] + 200
    g = ndi.gaussian_filter(sub, (2, 1, 1))
    R = hessian_matrix_eigvals(hessian_matrix(g, sigma=1.5, order="rc", use_gaussian_derivatives=True))[0]
    cr = ndi.gaussian_filter(sub, 2) < 150
    lab = cc3d.connected_components(cr); cnt = np.bincount(lab.ravel()); cnt[0] = 0
    cr = ndi.binary_fill_holes(lab == cnt.argmax())
    tissue = ndi.gaussian_filter(sub, 3) < 192
    dcr, ind = ndi.distance_transform_edt(~cr, return_indices=True)
    zone = (dcr > 0) & (dcr <= DMAX) & tissue
    surf, sflat, fi, (pg, pr) = profiles(cr, [g, R], zone, dcr, ind)
    pg = np.nan_to_num(pg, nan=255)
    pr = np.nan_to_num(pr, nan=0)
    out_band = (pg[:, DMIN:] > 171.5) | (pg[:, DMIN:] < 156)
    b = np.where(out_band.any(1), out_band.argmax(1) + DMIN, DMAX)
    # membrane evidence: a ridge peak within b-2 .. b+3 of the grey step
    ok = np.zeros(b.size, bool)
    for j in range(b.size):
        lo_, hi_ = max(b[j] - 2, 0), min(b[j] + 4, DMAX)
        ok[j] = pr[j, lo_:hi_].max() > R_THR
    b = np.where(ok & (b >= 4), b, 0).astype(np.float32)
    B = np.zeros(cr.size, np.float32); B[sflat] = b
    B = B.reshape(cr.shape)
    B2 = (ndi.gaussian_filter(B, SIG / 2) / (ndi.gaussian_filter(surf.astype(np.float32), SIG / 2) + 1e-6)).ravel()
    bvox = np.zeros(cr.shape, np.float32); bvox[zone] = B2[fi]
    lum = zone & (dcr <= bvox) & (bvox >= 3)
    vac = lum | ((edt.edt(~(lum | cr)) <= 2) & ~cr & tissue)
    # dense material touching the crystal (beyond the edge blur)
    g15 = ndi.gaussian_filter(sub, (2, 1.5, 1.5))
    cand = (g15 < DENSE) & tissue & (dcr >= 3) & (dcr <= DMAX)
    cand = ndi.binary_opening(cand, iterations=1)
    ll = cc3d.connected_components(cand, connectivity=6)
    ids = np.unique(ll[cand & (dcr <= 5)]); ids = ids[ids > 0]
    cs = np.bincount(ll.ravel())
    ids = ids[cs[ids] > 1000]
    dense = np.isin(ll, ids)
    print(name, "vac-with-membrane surface frac", round(float((b > 0).mean()), 3), "median thick px", np.median(b[b > 0]) if (b > 0).any() else 0,
          "| dense vox", int(dense.sum()))
    k = 20
    rgb = np.stack([np.clip(sub[k], 0, 255)] * 3, -1).astype(np.uint8)
    o = rgb.copy()
    o[vac[k]] = (0.5 * o[vac[k]] + 0.5 * np.array((240, 228, 66))).astype(np.uint8)
    o[dense[k]] = (0.5 * o[dense[k]] + 0.5 * np.array((0, 158, 115))).astype(np.uint8)
    o[find_boundaries(cr[k])] = (0, 0, 160)
    Image.fromarray(np.hstack([rgb, o])).save(f"{S}/vac2_{name}.png")


test("run0110", 407, slice(200, 580), slice(1060, 1520), "0110")
test("run0910", 279, slice(300, 700), slice(250, 1300), "0910")
test("run0910", 300, slice(0, 800), slice(2100, 2900), "0910tip")
