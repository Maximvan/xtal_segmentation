"""06 - QC overlay images for one run (10 nm if 07 has run, else 20 nm).

10 nm mode: thin outlines on the normalised bin2 image, so you can judge whether they sit on the lines.
Colours: cell outline per cell (orange / sky blue / green ...), crystal dark blue, nucleus magenta,
crystal vacuole yellow, mitochondria vermillion, dense granules black, dense material on crystal green.
usage: python 06_qc_overlays.py run0910|run0110
Output: 03_Output/<run>/qc/*.png
"""
import sys, json
import numpy as np, tifffile
from PIL import Image
from skimage.segmentation import find_boundaries
from xtal_lib import load, proc_dir, out_dir, RUNS

run = sys.argv[1]
D, O = proc_dir(run), out_dir(run)
Q = O / "qc"
Q.mkdir(exist_ok=True)
for f in Q.glob("*.png"):
    f.unlink()
CELLC = np.array([(0, 0, 0), (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (204, 121, 167),
                  (213, 94, 0), (150, 150, 150), (120, 80, 160)], np.uint8)

if (D / "lab10.npz").exists():
    c = json.load(open(D / "crop10.json"))
    lo, n = c["lo10"], c["shape10"]
    L = load(D / "lab10")
    vac = load(D / "vac10").astype(bool)
    mi = load(D / "mito10") > 0
    gr = load(D / "gran10").astype(bool) if (D / "gran10.npz").exists() else np.zeros_like(mi)
    dn = load(D / "dense10").astype(bool) if (D / "dense10.npz").exists() else np.zeros_like(mi)
    raw = tifffile.memmap(RUNS[run]["bin2"])
    norm = json.load(open(D / "norm.json"))
    resin, gain = np.array(norm["resin"]), norm["gain"]

    def img(sl):  # sl in crop coords
        sl = tuple(slice(*s.indices(nn)) if isinstance(s, slice) else s for s, nn in zip(sl, n))
        full = tuple(slice(s.start + o, s.stop + o) if isinstance(s, slice) else s + o for s, o in zip(sl, lo))
        a = raw[full].astype(np.float32)
        zi = np.arange(lo[0], lo[0] + n[0])[sl[0]] // 2
        rz = resin[np.minimum(zi, len(resin) - 1)]
        if np.ndim(rz) == 0:
            a = a - rz
        else:  # reslice: Z is the first remaining axis
            a = a - rz[:, None]
        return np.clip(a * gain + 200, 0, 255).astype(np.uint8)

    def ov(sl):
        im = img(sl)
        l, va, m = L[sl], vac[sl], mi[sl]
        rgb = np.stack([im] * 3, -1)
        o = rgb.copy()
        cid = np.where(l >= 30, l - 30, np.where(l >= 10, l - 10, 0))
        o[va & (l != 2)] = (0.5 * o[va & (l != 2)] + 0.5 * np.array((240, 228, 66))).astype(np.uint8)
        b = find_boundaries(cid, mode="inner") & (cid > 0)
        o[b] = CELLC[np.minimum(cid[b], len(CELLC) - 1)]
        o[find_boundaries(m, mode="inner")] = (213, 94, 0)
        o[find_boundaries(gr[sl], mode="inner")] = (0, 0, 0)
        o[find_boundaries(dn[sl], mode="inner")] = (0, 158, 115)
        o[find_boundaries(l >= 30, mode="inner")] = (255, 0, 255)
        o[find_boundaries(l == 2, mode="inner")] = (0, 0, 160)
        o[find_boundaries(va, mode="outer") & (l != 2)] = (200, 180, 0)
        return rgb, o

    for z in np.linspace(n[0] * 0.1, n[0] * 0.9, 8).astype(int):
        rgb, o = ov(np.s_[z, :, :])
        Image.fromarray(np.vstack([rgb, o])).save(Q / f"xy10_z{z + lo[0]:04d}.png")
    cr = L == 2
    y = int(np.where(cr.any((0, 2)))[0].mean())
    Image.fromarray(np.vstack(ov(np.s_[:, y, :]))).save(Q / f"xz10_y{y + lo[1]}.png")
    x = int(np.where(cr.any((0, 1)))[0].mean())
    Image.fromarray(np.vstack(ov(np.s_[:, :, x]))).save(Q / f"zy10_x{x + lo[2]}.png")
else:
    v = tifffile.imread(D / "img20.tif")
    cr = load(D / "crystal_refined").astype(bool)
    nu, ce = load(D / "nuclei"), load(D / "cells")
    vx = load(D / "vac_xtal").astype(bool)
    mi = load(D / "mito") > 0

    def ov(sl):
        im, c, nn, cl, a, m_ = [x[sl] for x in (v, cr, nu, ce, vx, mi)]
        rgb = np.stack([im] * 3, -1).astype(np.float32)
        lab = np.minimum(cl, len(CELLC) - 1)
        t = cl > 0
        rgb[t] = 0.65 * rgb[t] + 0.35 * CELLC[lab[t]]
        rgb[a & ~c] = 0.4 * rgb[a & ~c] + 0.6 * np.array((240, 228, 66))
        rgb[find_boundaries(c)] = (0, 0, 160)
        rgb[find_boundaries(nn > 0)] = (255, 0, 255)
        rgb[m_] = (213, 94, 0)
        return rgb.astype(np.uint8)

    for z in np.linspace(v.shape[0] * 0.1, v.shape[0] * 0.9, 8).astype(int):
        Image.fromarray(np.hstack([np.stack([v[z]] * 3, -1), ov(np.s_[z])])).save(Q / f"xy_z{z:04d}.png")
print("qc written", Q)
