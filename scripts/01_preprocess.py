"""01 - Preprocess a run: bin4 stack -> 20 nm isotropic, normalised, valid-data mask.

Normalisation: per-slice OFFSET (resin peak -> 200) and one GLOBAL GAIN per run (crystal peak -> 130).
Raw contrast is stable along Z, while per-slice percentile anchors drift with how much crystal
is in the slice, so only the offset is fitted per slice.

usage: python 01_preprocess.py run0910|run0110
Outputs in 02_Processed/<run>/: img20.tif (uint8), valid.npy (bool), norm.json
"""
import sys, json
import numpy as np, tifffile
from scipy import ndimage as ndi
from xtal_lib import RUNS, proc_dir

run = sys.argv[1]
OUT = proc_dir(run)
raw = tifffile.memmap(RUNS[run]["bin4"])
Z = raw.shape[0] // 2

# pass 1: per-slice resin peak + pooled histogram of (value - resin) for the crystal peak
resin = np.zeros(Z)
pool = np.zeros(256)
for z in range(Z):
    s = raw[2 * z:2 * z + 2].astype(np.float32).mean(0)
    val = (raw[2 * z] > 0) & (raw[2 * z] < 255) & (raw[2 * z + 1] > 0) & (raw[2 * z + 1] < 255)
    sm = ndi.gaussian_filter(s, 3)[val]
    h, e = np.histogram(sm, 256, (0, 256))
    resin[z] = e[np.argmax(h)] + 0.5
    d = np.clip(np.round(resin[z] - sm), 0, 255).astype(int)
    pool += np.bincount(d, minlength=256)[:256]
resin = ndi.median_filter(resin, 9, mode="nearest")
win = np.arange(85, 141)
diff = win[np.argmax(pool[win])]  # resin - crystal distance (dark, large, homogeneous body)
gain = 70.0 / diff
print(run, "crystal at resin -", diff, "gain", round(gain, 3), flush=True)

img = np.empty((Z, *raw.shape[1:]), np.uint8)
valid = np.empty((Z, *raw.shape[1:]), bool)
for z in range(Z):
    a, b = raw[2 * z].astype(np.float32), raw[2 * z + 1].astype(np.float32)
    pad = (a >= 254.5) | (a <= 0.5) | (b >= 254.5) | (b <= 0.5)
    valid[z] = ~ndi.binary_opening(pad, iterations=1)
    n = ((a + b) / 2 - resin[z]) * gain + 200
    n[~valid[z]] = 200
    img[z] = np.clip(n, 0, 255).astype(np.uint8)

tifffile.imwrite(OUT / "img20.tif", img, imagej=True,
                 resolution=(1 / 0.02, 1 / 0.02), metadata={"spacing": 0.02, "unit": "micron"})
from xtal_lib import save
save(OUT / "valid", valid)
json.dump(dict(gain=gain, crystal_offset=int(diff), resin=resin.tolist()), open(OUT / "norm.json", "w"))
print("done", img.shape)
