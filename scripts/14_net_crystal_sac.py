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
# light 3D consistency, remove specks, fill holes
cry = ndi.binary_opening(cry, structure=np.ones((3, 3, 3)))
lab = cc3d.connected_components(cry, connectivity=6)
cnt = cc3d.statistics(lab)["voxel_counts"]
keep = np.where(cnt >= MIN_UM3 * 1e6)[0]; keep = keep[keep > 0]
cry = fill_holes3d(np.isin(lab, keep))
del lab
log("crystals kept", len(keep), "volumes um3", sorted([round(float(c) * 1e-6, 2) for c in cnt[keep]], reverse=True)[:10])
sac &= ~cry
save(D / "net_crystal10", cry)
save(D / "net_sac10", sac)
# 20 nm grid for 02: mean over 2x2x2 >= 0.5, padded/cropped to img20's shape
img20 = tifffile.memmap(D / "img20.tif")
z2, h2, w2 = Z // 2, H // 2, W // 2
c20 = cry[:2 * z2, :2 * h2, :2 * w2].reshape(z2, 2, h2, 2, w2, 2).mean((1, 3, 5)) >= 0.5
out = np.zeros(img20.shape, bool)
s = [min(a, b) for a, b in zip(img20.shape, c20.shape)]
out[:s[0], :s[1], :s[2]] = c20[:s[0], :s[1], :s[2]]
save(D / "crystal_net", out)
log("done", "crystal um3", round(cry.sum() * 1e-6, 1), "sac um3", round(sac.sum() * 1e-6, 1))
