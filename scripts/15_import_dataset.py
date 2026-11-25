"""15 - Import a new FIB-SEM stack: resample to the pipeline grids (bin2-equivalent 10 nm isotropic,
bin4-equivalent 20 x 20 x 10 nm), written as contiguous ImageJ TIFFs in 01_Input/derived/.

Z is resampled with area weighting (each output slice = weighted mean of the input slices it covers, so
noise averages out when going 6/8 nm -> 10 nm); XY with linear interpolation (up) / area averaging (down).
usage: python 15_import_dataset.py <name> <source tif> <xy nm> <z nm>
"""
import sys
import numpy as np, tifffile, cv2
from xtal_lib import ROOT

name, src, xy, zn = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
OUT = ROOT / "01_Input/derived"; OUT.mkdir(parents=True, exist_ok=True)
st = tifffile.memmap(src)
Z, H, W = st.shape
T = 10.0
nz = int(np.floor(Z * zn / T))
H10, W10 = int(round(H * xy / T)), int(round(W * xy / T))
H10 -= H10 % 2; W10 -= W10 % 2
interp = cv2.INTER_LINEAR if xy < T else cv2.INTER_AREA
print(name, "source", st.shape, f"{xy}x{xy}x{zn} nm -> 10 nm", (nz, H10, W10), flush=True)
p10 = OUT / f"{name}_10nm.tif"
p20 = OUT / f"{name}_20nm.tif"
o10 = tifffile.memmap(p10, shape=(nz, H10, W10), dtype=np.uint8, imagej=True,
                      metadata=dict(spacing=0.01, unit="micron"), resolution=(100, 100))
o20 = tifffile.memmap(p20, shape=(nz, H10 // 2, W10 // 2), dtype=np.uint8, imagej=True,
                      metadata=dict(spacing=0.01, unit="micron"), resolution=(50, 50))
r = T / zn  # input slices per output slice
for k in range(nz):
    a, b = k * r, (k + 1) * r
    acc = np.zeros((H, W), np.float32); wsum = 0.0
    for i in range(int(np.floor(a)), min(int(np.ceil(b)), Z)):
        w = min(b, i + 1) - max(a, i)
        if w > 0:
            acc += w * np.asarray(st[i], np.float32); wsum += w
    sl = acc / max(wsum, 1e-6)
    s10 = cv2.resize(sl, (W10, H10), interpolation=interp)
    o10[k] = np.clip(np.rint(s10), 0, 255).astype(np.uint8)
    o20[k] = np.clip(np.rint(cv2.resize(s10, (W10 // 2, H10 // 2), interpolation=cv2.INTER_AREA)), 0, 255).astype(np.uint8)
    if k % 200 == 0:
        print(k, "/", nz, flush=True)
o10.flush(); o20.flush()
del o10, o20
print("written", p10, p20, flush=True)
