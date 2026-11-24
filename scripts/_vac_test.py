"""scratch: lumen between crystal and the first membrane ridge, on small 10 nm crops."""
import sys, json
import numpy as np, tifffile, cc3d, edt
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.segmentation import find_boundaries
from xtal_lib import RUNS

S = sys.argv[1]
thr = float(sys.argv[2])


def radial_lumen(cr, R, tissue, thr, dmax=30, dmin=2, sig=4.0, g=None):
    """vacuole edge = first outward step from vacuole grey (~165) to cytoplasm grey (~177), per crystal-surface
    voxel, using shell-mean profiles averaged over neighbouring surface voxels (sigma `sig` px)."""
    dcr, ind = ndi.distance_transform_edt(~cr, return_indices=True)
    surf = cr & ~ndi.binary_erosion(cr, np.ones((3, 3, 3)), border_value=1)
    zone = (dcr > 0) & (dcr <= dmax) & tissue
    iz, iy, ix = ind[0][zone], ind[1][zone], ind[2][zone]
    d = np.rint(dcr[zone]).astype(np.int64)
    gv = g[zone]
    w = ndi.gaussian_filter(surf.astype(np.float32), sig) + 1e-6
    prof = []
    for k in range(1, dmax + 1):
        m = d == k
        Sx = np.zeros(cr.shape, np.float32); Cx = np.zeros(cr.shape, np.float32)
        np.add.at(Sx, (iz[m], iy[m], ix[m]), gv[m]); np.add.at(Cx, (iz[m], iy[m], ix[m]), 1)
        Ss, Cs = ndi.gaussian_filter(Sx, sig), ndi.gaussian_filter(Cx, sig)
        prof.append(np.where(Cs[surf] > 0.05, Ss[surf] / np.maximum(Cs[surf], 1e-6), 255))
    prof = np.stack(prof, 1)  # (n_surf, dmax): mean grey at distance k+1
    hit = prof[:, dmin:] > thr
    b = np.where(hit.any(1), hit.argmax(1) + dmin, dmax).astype(np.float32)
    B = np.zeros(cr.shape, np.float32); B[surf] = b
    B2 = ndi.gaussian_filter(B, sig / 2) / (ndi.gaussian_filter(surf.astype(np.float32), sig / 2) + 1e-6)
    bvox = np.zeros(cr.shape, np.float32)
    bvox[zone] = B2[iz, iy, ix]
    lum = zone & (dcr <= bvox)
    print("  thickness px median", np.median(b), "p10", np.percentile(b, 10), "p90", np.percentile(b, 90))
    return lum, bvox


def test(run, z20, ys, xs, name):
    raw = tifffile.memmap(RUNS[run]["bin2"])
    n = json.load(open(f"../02_Processed/{run}/norm.json"))
    z = 2 * z20
    zr = np.arange(z - 20, z + 21)
    sub = raw[zr[0]:zr[-1] + 1, ys, xs].astype(np.float32)
    sub = (sub - np.array(n["resin"])[zr // 2][:, None, None]) * n["gain"] + 200
    g = ndi.gaussian_filter(sub, 1)
    R = hessian_matrix_eigvals(hessian_matrix(sub, sigma=1.5, order="rc", use_gaussian_derivatives=True))[0]
    cr = ndi.gaussian_filter(sub, 2) < 150
    lab = cc3d.connected_components(cr); cnt = np.bincount(lab.ravel()); cnt[0] = 0
    cr = ndi.binary_fill_holes(lab == cnt.argmax())
    dcr = edt.edt(~cr)
    tissue = ndi.gaussian_filter(sub, 3) < 192
    lum, bmap = radial_lumen(cr, R, tissue, thr, g=ndi.gaussian_filter(sub, 1))
    vac = lum | ((edt.edt(~(lum | cr)) <= 3) & ~cr & tissue)
    print(name, "lumen vox", lum.sum(), "mean width px", lum.sum() / max((cr & (dcr == 0) & (edt.edt(cr) <= 1)).sum(), 1))
    k = 20
    rgb = np.stack([np.clip(g[k], 0, 255)] * 3, -1).astype(np.uint8)
    o = rgb.copy()
    o[vac[k] & ~lum[k]] = (240, 228, 66)
    o[lum[k]] = (86, 180, 233)
    o[find_boundaries(cr[k])] = (0, 0, 160)
    r = (np.clip(R[k] / 12, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(np.hstack([rgb, o, np.stack([r] * 3, -1)])).save(f"{S}/vac_{name}.png")


test("run0110", 407, slice(200, 580), slice(1060, 1520), "0110")
test("run0910", 279, slice(300, 700), slice(250, 1300), "0910")
