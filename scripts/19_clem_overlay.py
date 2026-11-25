"""19 - 3D overlay of the CLEM light-microscopy channels on the smallXtal_6M EM reconstruction.

Overlay_test1 binned.tif (873 slices x 5 ch x 950 x 781, 32 nm px) is the LM registered onto the EM, but in
block-face orientation: overlay slice = EM depth (y), rows = EM milling direction (z), cols = EM x, and both
rows and cols are flipped. Registration to the 6M 20 nm grid (template matching of the EM channel, 2026-11-25,
NCC 0.86-0.92): y20 = 0.4 * overlay_slice + 72, z10 = 3.2 * (949 - row) - 2, x20 = 1.6 * (780 - col).
Channels (user): 0 FITC, 1 DAPI, 2 EM/brightfield, 3 + 4 NET stainings. Each LM channel is resampled onto a
40 nm grid in EM coordinates, thresholded (max of median + 5 MAD and 30 % of the 99.9th percentile), small blobs removed, meshed (marching cubes), and
written as PLY in 03_Output/smallXtal_6M/clem/ (10 adds them to the 3D scene).
usage: python 19_clem_overlay.py
"""
import numpy as np, tifffile, trimesh, cc3d
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import fast_simplification as fs
from xtal_lib import ROOT, out_dir

OV = ROOT / "01_Input/FibSEM xtals/EM447_CLEM2_smallXtal_6M_0408/Overlay_test1 binned.tif"
O = out_dir("smallXtal_6M") / "clem"; O.mkdir(exist_ok=True)
ov = tifffile.memmap(OV)  # (873, 5, 950, 781)
NZ, NC, NR, NCOL = ov.shape
d20 = tifffile.memmap(ROOT / "01_Input/derived/smallXtal_6M_20nm.tif")  # (Z10, Y20, X20)
Z10, Y20, X20 = d20.shape
# target grid: 40 nm in all axes -> indices z10 = 4i, y20 = 2j, x20 = 2k
zi, yj, xk = np.arange(0, Z10, 4), np.arange(0, Y20, 2), np.arange(0, X20, 2)
sl = (yj - 72) / 0.4                       # overlay slice
row = 949 - (zi + 2) / 3.2                 # overlay row (flipped)
col = 780 - xk / 1.6                       # overlay col (flipped)
okz = (sl >= 0) & (sl <= NZ - 1)
print("EM y range covered by LM (20 nm px):", yj[okz].min(), yj[okz].max())
CH = {0: ("FITC", (0, 158, 115)), 1: ("DAPI", (86, 180, 233)), 3: ("NET_stain_1", (204, 121, 167)),
      4: ("NET_stain_2", (230, 159, 0))}
J, I, K = np.meshgrid(sl, row, col, indexing="ij")  # (y, z, x) order of the overlay: slice, row, col
for c, (name, rgb) in CH.items():
    vol = np.asarray(ov[:, c]).astype(np.float32)
    v = ndi.map_coordinates(vol, [J, I, K], order=1, mode="constant", cval=0)  # (nY, nZ, nX)
    v = np.transpose(v, (1, 0, 2))  # -> (z, y, x) on the 40 nm grid
    v = ndi.gaussian_filter(v, 1.0)
    med = np.median(v[v > 0]) if (v > 0).any() else 0
    mad = np.median(np.abs(v[v > 0] - med)) + 1e-3
    thr = max(med + 5 * mad, 0.3 * float(np.percentile(v, 99.9)))  # sparse channels: MAD ~ 0, use the bright end
    m = v > thr
    lab = cc3d.connected_components(m, connectivity=26)
    cnt = cc3d.statistics(lab)["voxel_counts"]
    keep = np.where(cnt >= 30)[0]; keep = keep[keep > 0]   # >= ~0.002 um3
    m = np.isin(lab, keep)
    print(name, "threshold", round(float(thr), 1), "voxels", int(m.sum()), "blobs", len(keep))
    if m.sum() < 50:
        continue
    f = np.pad(ndi.gaussian_filter(m.astype(np.float32), 0.7), 1)
    vv, ff, _, _ = marching_cubes(f, 0.5)
    vv = (vv - 1) * 4  # 40 nm grid -> 10 nm index units (z10, y10, x10)
    vv = vv[:, ::-1] * 0.01  # (x, y, z) in um, same frame as 05/10 meshes
    vv[:, 1] *= -1
    if len(ff) > 40000:
        vv, ff = fs.simplify(vv.astype(np.float32), ff.astype(np.int32), target_reduction=1 - 40000 / len(ff))
    me = trimesh.Trimesh(vv, ff, process=True)
    trimesh.smoothing.filter_taubin(me, iterations=5)
    me.visual.face_colors = [*rgb, 255]
    me.export(O / f"{name}.ply")
print("written", sorted(p.name for p in O.glob("*.ply")))
