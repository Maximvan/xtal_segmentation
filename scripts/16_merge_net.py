"""16 - Put the U-Net crystal + sac (14) into the 10 nm labels of 07.

  crystal : network crystal replaces label 2 inside the crop; voxels that were crystal but are not any more
            get the most common neighbouring non-crystal label (iterative grey dilation)
  sac     : network sac, inside cells, connected to a crystal -> vac10 (then 12 applies the coverage rule)
usage: python 16_merge_net.py <run>
"""
import sys, json, time
import numpy as np, cc3d
from scipy import ndimage as ndi
from xtal_lib import load, save, proc_dir

run = sys.argv[1]
D = proc_dir(run)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
c = json.load(open(D / "crop10.json")); lo, n = c["lo10"], c["shape10"]
sl = tuple(slice(l, l + k) for l, k in zip(lo, n))


def crop(a):
    a = a[sl]
    return np.pad(a, [(0, k - s) for k, s in zip(n, a.shape)])


L = load(D / "lab10")
cr = crop(load(D / "net_crystal10").astype(bool))
sac = crop(load(D / "net_sac10").astype(bool))
old = L == 2
log("crystal vox old", int(old.sum()), "net", int(cr.sum()))
# voxels that stop being crystal: take the neighbouring label (resin / cell / nucleus / other tissue)
lost = old & ~cr
L[lost] = 0
for _ in range(30):
    if not (L[lost] == 0).any():
        break
    g = ndi.grey_dilation(np.where(L == 2, 0, L), size=3)
    fill = lost & (L == 0)
    L[fill] = g[fill]
L[lost & (L == 0)] = 1
L[cr] = 2
save(D / "lab10", L)
# sac: inside cells, connected to the crystal
cell = L >= 10
sac &= cell & ~cr
lab = cc3d.connected_components(sac | cr, connectivity=6)
ids = np.unique(lab[cr]); ids = ids[ids > 0]
sac = np.isin(lab, ids) & ~cr
del lab
if (D / "vac10_unconstrained.npz").exists():
    (D / "vac10_unconstrained.npz").unlink()  # 12 must start from this new sac
save(D / "vac10", sac)
log("done: crystal um3", round(cr.sum() * 1e-6, 2), "sac um3", round(sac.sum() * 1e-6, 2))
