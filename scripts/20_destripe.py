"""20 - Destripe a run: per-slice two-point intensity correction (resin + crystal), the cause of stripes.

FIB-SEM detector contrast drifts from slice to slice: the resin level is stable but dark material (crystal,
chromatin) jumps by ~2-5 grey values between neighbouring slices, so every threshold / per-slice step near a
boundary flips and the 3D surfaces get stripes along the slices. Here each slice is linearly rescaled so that
its resin level and its crystal level (inside the U-Net crystal mask, eroded) match the stack medians:
    v' = r_ref + (v - r_z) * (r_ref - c_ref) / (r_z - c_z)
Slices without enough crystal get c_z interpolated from the neighbours. Writes destriped bin2 (10 nm) and bin4
(20 nm) stacks to 01_Input/derived/<run>_ds_{10,20}nm.tif; xtal_lib.RUNS uses them when present.
usage: python 20_destripe.py <run>
"""
import sys, json
import numpy as np, tifffile
from scipy import ndimage as ndi
from xtal_lib import RUNS, ROOT, load, proc_dir

run = sys.argv[1]
D = proc_dir(run)
src2 = RUNS[run]["bin2_orig"] if "bin2_orig" in RUNS[run] else RUNS[run]["bin2"]
src4 = RUNS[run]["bin4_orig"] if "bin4_orig" in RUNS[run] else RUNS[run]["bin4"]
s2, s4 = tifffile.memmap(src2), tifffile.memmap(src4)
Z = s2.shape[0]
assert s4.shape[0] == Z, (s2.shape, s4.shape)
cr = load(D / "net_crystal10").astype(bool)
r = np.zeros(Z); c = np.full(Z, np.nan)
for z in range(Z):
    a = np.asarray(s2[z])
    v = a[::2, ::2].ravel(); v = v[(v > 2) & (v < 253)]
    h = ndi.gaussian_filter1d(np.bincount(v, minlength=256).astype(float), 1.5)
    m = int(np.argmax(h))
    near = v[np.abs(v.astype(int) - m) <= 8]
    r[z] = near.mean() if near.size else m          # sub-grey resin level
    if z < cr.shape[0]:
        k = cr[z, :a.shape[0], :a.shape[1]]
        if k.sum() > 20000:
            k = ndi.binary_erosion(k, iterations=8)
            if k.sum() > 5000:
                c[z] = np.median(a[k[:a.shape[0], :a.shape[1]]])
ok = ~np.isnan(c)
print(run, "slices with crystal reference", int(ok.sum()), "/", Z, flush=True)
if ok.sum() < 10:
    raise SystemExit("not enough crystal to destripe")
zi = np.arange(Z)
c = np.interp(zi, zi[ok], c[ok])
c = ndi.median_filter(c, 3)  # remove single-slice measurement outliers only (jitter itself is kept -> corrected)
r_ref, c_ref = np.median(r), np.median(c)
g = (r_ref - c_ref) / np.maximum(r - c, 5)
print("contrast factor per slice: median %.3f  p5 %.3f  p95 %.3f" % (np.median(g), *np.percentile(g, [5, 95])), flush=True)
OUT = ROOT / "01_Input/derived"; OUT.mkdir(parents=True, exist_ok=True)
for src, s, tag in ((src2, s2, "10nm"), (src4, s4, "20nm")):
    p = OUT / f"{run}_ds_{tag}.tif"
    o = tifffile.memmap(p, shape=s.shape, dtype=np.uint8, imagej=True, metadata=dict(spacing=0.01, unit="micron"))
    for z in range(Z):
        a = np.asarray(s[z]).astype(np.float32)
        valid = (a > 0.5) & (a < 254.5)   # keep empty canvas / saturated pixels as they are
        b = r_ref + (a - r[z]) * g[z]
        o[z] = np.where(valid, np.clip(np.rint(b), 1, 254), a).astype(np.uint8)
    o.flush(); del o
    print("written", p, flush=True)
json.dump(dict(r=r.tolist(), c=c.tolist(), g=g.tolist(), r_ref=float(r_ref), c_ref=float(c_ref)),
          open(D / "destripe.json", "w"))
