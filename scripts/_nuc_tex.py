"""scratch: texture-based nuclei (smooth + mid-grey cores, boundary snapped to the envelope by watershed)."""
import sys, time
import numpy as np, tifffile, cc3d, edt
from PIL import Image
from skimage.segmentation import watershed, find_boundaries
from xtal_lib import load, save, proc_dir, gpu_filter, gauss, dilate, erode, eopen, eclose, fill_holes3d, keep_large, VX, NT
from cupyx.scipy import ndimage as cndi
import cupy as cp

run = sys.argv[1]
D = proc_dir(run)
t0 = time.time()
v = tifffile.imread(D / "img20.tif")
cr = load(D / "crystal").astype(bool)
tis = load(D / "tissue").astype(bool) & ~cr


def sdf(a):
    a = cndi.gaussian_filter(a.astype("float32"), 1)
    m = cndi.gaussian_filter(a, 2)
    return cp.sqrt(cp.maximum(cndi.gaussian_filter(a * a, 2) - m * m, 0))


sd = gpu_filter(v, sdf, halo=12)
tex = gauss(sd, 3)
med = gpu_filter(v, lambda a: cndi.median_filter(a, size=5), halo=4, chunk=64, out_dtype=np.uint8)
print("features", round(time.time() - t0), flush=True)

core = (tex < float(sys.argv[2]) if len(sys.argv) > 2 else tex < 2.5) & (med > 145) & (med < 178) & tis & ~dilate(cr, 5)
core = eopen(core, 4)
core = keep_large(core, 40000)  # > 0.32 um3
print("cores", round(core.sum() * VX**3, 1), cc3d.connected_components(core).max(), flush=True)

# watershed on the edge map: nucleus marker = core, background marker = far from cores or resin/crystal
elev = gauss(sd, 0.7)
mk = np.zeros(v.shape, np.uint8)
mk[erode(core, 1)] = 1
br = gpu_filter(v, lambda a: cndi.maximum_filter(cndi.gaussian_filter(a.astype('float32'), 1), size=(1, 7, 7)), halo=4)
mk[~dilate(core, 14) | ~tis | (((br > float(sys.argv[4])) | (med > 182)) & ~dilate(core, 3))] = 2
zz, yy, xx = np.where(dilate(core, 15))
bb = np.s_[zz.min():zz.max() + 1, yy.min():yy.max() + 1, xx.min():xx.max() + 1]
lab = watershed(elev[bb], markers=mk[bb])
nu = np.zeros(v.shape, bool)
nu[bb] = lab == 1
nu = fill_holes3d(nu)
nu = eopen(nu, 2)
nu = keep_large(nu, 125000)
save(D / "nuclei_tex_raw", nu)
from scipy import ndimage as ndi
nc = eclose(nu, 2)
for ax in range(3):  # light-chromatin patches enclosed in any XY/XZ/YZ section belong to the nucleus
    f = np.moveaxis(nc, ax, 0).copy()
    for i in range(f.shape[0]):
        if f[i].any():
            f[i] = ndi.binary_fill_holes(f[i])
    nu |= np.moveaxis(f, 0, ax) & tis
nu = fill_holes3d(nu)
print("nuclei", round(nu.sum() * VX**3, 1), "um3", [round(c * VX**3, 1) for c in cc3d.statistics(cc3d.connected_components(nu))["voxel_counts"][1:]], round(time.time() - t0), flush=True)
save(D / "nuclei_tex", nu)

old = load(D / "nuclei") > 0
S = sys.argv[3] if len(sys.argv) > 3 else "."
for z in np.linspace(v.shape[0] * 0.1, v.shape[0] * 0.9, 6).astype(int):
    rgb = np.stack([v[z]] * 3, -1)
    a, b = rgb.copy(), rgb.copy()
    a[find_boundaries(old[z])] = (255, 0, 255)
    b[find_boundaries(nu[z])] = (255, 0, 255)
    Image.fromarray(np.vstack([a, b])[:, 100:1400]).save(f"{S}/nuc_z{z:03d}.png")
