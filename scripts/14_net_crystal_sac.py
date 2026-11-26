"""14 - Apply the crystal/sac U-Net (13, trained on Jasmijn's annotations) to a whole run at 10 nm.

Writes (02_Processed/<run>/):
  net_crystal10.npz  bool, full bin2 grid: crystal bodies (> MIN_UM3, 3D holes filled)
  net_sac10.npz      bool, full bin2 grid: raw sac prediction (cleaned later with the cells: 16)
  crystal_net.npz    bool, 20 nm grid (img20): used by 02 instead of the threshold crystal
usage: CUDA_VISIBLE_DEVICES=5 python 14_net_crystal_sac.py <run>
"""
import sys, json, time
import numpy as np, tifffile, cc3d
from scipy import ndimage as ndi
from xtal_lib import RUNS, ROOT, save, proc_dir, fill_holes3d
import importlib
m13 = importlib.import_module("13_sac_unet")

run = sys.argv[1]
D = proc_dir(run)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
MIN_UM3 = 0.02  # smallest crystal kept (20 000 voxels at 10 nm)

net = m13.load_net()
raw = tifffile.memmap(RUNS[run]["bin2"])
Z, H, W = raw.shape
cry = np.zeros((Z, H, W), bool)
sac = np.zeros((Z, H, W), bool)
for z, X in m13.run_slices(run, range(Z)):
    p = m13.predict(net, X)
    am = p.argmax(0)
    cry[z] = am == 1
    sac[z] = am == 2
    if z % 200 == 0:
        log("slice", z, "/", Z, "crystal px", int(cry[z].sum()), "sac px", int(sac[z].sum()))
def down2(m):
    """2x2x2 majority downsample, slice pair by slice pair (no huge float intermediates)"""
    z2, h2, w2 = m.shape[0] // 2, m.shape[1] // 2, m.shape[2] // 2
    out = np.zeros((z2, h2, w2), bool)
    for k in range(z2):
        a = m[2 * k, :2 * h2, :2 * w2].astype(np.uint8) + m[2 * k + 1, :2 * h2, :2 * w2]
        out[k] = a.reshape(h2, 2, w2, 2).sum((1, 3)) >= 4
    return out


# remove specks: decide which crystal bodies to keep on a 2x coarser grid (memory-light for 10 GVox runs),
# then apply at 10 nm; fill holes slice by slice
z2, h2, w2 = Z // 2, H // 2, W // 2
c2 = down2(cry)
c2 = ndi.binary_opening(c2, structure=np.ones((3, 3, 3)))
lab = cc3d.connected_components(c2, connectivity=6)
cnt = cc3d.statistics(lab)["voxel_counts"]
keep = np.where(cnt * 8 >= MIN_UM3 * 1e6)[0]; keep = keep[keep > 0]
k2 = np.isin(lab, keep)
del lab, c2
k2 = ndi.binary_dilation(k2, iterations=2)
for z in range(2 * z2):  # apply the keep-mask slice by slice
    cry[z, :2 * h2, :2 * w2] &= k2[z // 2].repeat(2, 0).repeat(2, 1)
cry[2 * z2:] = False
del k2
for z in range(Z):
    if cry[z].any():
        cry[z] = ndi.binary_fill_holes(cry[z])
log("crystals kept", len(keep), "volumes um3 (20 nm grid)", sorted([round(float(c) * 8e-6, 2) for c in cnt[keep]], reverse=True)[:10])
sac &= ~cry
save(D / "net_crystal10", cry)
save(D / "net_sac10", sac)
# 20 nm grid for 02: mean over 2x2x2 >= 0.5, padded/cropped to img20's shape
img20 = tifffile.memmap(D / "img20.tif")
z2, h2, w2 = Z // 2, H // 2, W // 2
c20 = down2(cry)
out = np.zeros(img20.shape, bool)
s = [min(a, b) for a, b in zip(img20.shape, c20.shape)]
out[:s[0], :s[1], :s[2]] = c20[:s[0], :s[1], :s[2]]
save(D / "crystal_net", out)
log("done", "crystal um3", round(cry.sum() * 1e-6, 1), "sac um3", round(sac.sum() * 1e-6, 1))
