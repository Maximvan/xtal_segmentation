"""Shared config + helpers for the xtal_segmentation pipeline.

Runs are addressed by short name (run0910, run0110). Working grid = 20 nm isotropic
(bin4 XY, Z pairs averaged). Heavy filtering runs on the GPU (CuPy) in Z-chunks with a halo;
connected components (cc3d) and distance transforms (edt) run multithreaded on the CPU.
"""
from pathlib import Path
import numpy as np
import cc3d, edt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "01_Input/FibSEM xtals/2026_10_01_JasmijnB_FIBSEM_CLEM1"
RUNS = {
    "run0910": dict(bin4=RAW / "run_0910/EM447_CLEM1_0910_bin4.tif",
                    bin2=RAW / "run_0910/EM447_CLEM1_0910_bin2.tif"),
    "run0110": dict(bin4=RAW / "run_0110/EM_447_CLEM1_1001_bin4.tif",
                    bin2=RAW / "run_0110/EM_447_CLEM1_1001_bin2.tif"),
}
VX = 0.02  # um, working grid
NT = 16    # CPU threads


def proc_dir(run):
    d = ROOT / "02_Processed" / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir(run):
    d = ROOT / "03_Output" / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def normalise_slice(s, valid):
    """Map the resin peak to 200 and the 1st percentile to 60 (per slice)."""
    vals = s[valid]
    hist, e = np.histogram(vals, 256, (0, 256))
    resin = e[np.argmax(hist)] + 0.5
    lo = np.percentile(vals, 1)
    n = (s - resin) * (140 / max(resin - lo, 1)) + 200
    n[~valid] = 200
    return np.clip(n, 0, 255)


# ---------------- GPU filters (chunked over Z) ----------------
def gpu_filter(vol, fn, halo=16, chunk=96, out_dtype=np.float32):
    """Apply fn(cupy_array)->cupy_array per Z-chunk with halo; returns numpy array."""
    import cupy as cp
    Z = vol.shape[0]
    out = np.empty(vol.shape, out_dtype)
    for z0 in range(0, Z, chunk):
        z1 = min(Z, z0 + chunk)
        a0, a1 = max(0, z0 - halo), min(Z, z1 + halo)
        r = fn(cp.asarray(vol[a0:a1]))
        out[z0:z1] = cp.asnumpy(r[z0 - a0:z0 - a0 + (z1 - z0)]).astype(out_dtype, copy=False)
        del r
        cp.get_default_memory_pool().free_all_blocks()
    return out


def gauss(vol, sigma):
    from cupyx.scipy import ndimage as cndi
    return gpu_filter(vol, lambda a: cndi.gaussian_filter(a.astype("float32"), sigma),
                      halo=int(4 * sigma) + 2)


def local_std(vol, size):
    from cupyx.scipy import ndimage as cndi
    def f(a):
        a = a.astype("float32")
        m = cndi.uniform_filter(a, size)
        return (cndi.uniform_filter(a * a, size) - m * m).clip(0) ** 0.5
    return gpu_filter(vol, f, halo=size)


# ---------------- exact spherical morphology via EDT (CPU, multithreaded) ----------------
def dilate(m, r):
    return edt.edt(~m, parallel=NT, black_border=False) <= r if r > 0 else m


def erode(m, r):
    return edt.edt(m, parallel=NT, black_border=True) > r if r > 0 else m


def eopen(m, r):
    return dilate(erode(m, r), r)


def eclose(m, r):
    return erode(dilate(m, r), r)


def fill_holes3d(m):
    """Fill background components not touching the volume border."""
    lab = cc3d.connected_components(~m, connectivity=6)
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
                                       lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    return m | ~np.isin(lab, border)


def fill2d(m):
    from scipy import ndimage as ndi
    out = m.copy()
    for z in range(m.shape[0]):
        if m[z].any():
            out[z] = ndi.binary_fill_holes(m[z])
    return out


def keep_large(m, min_vox, connectivity=6):
    lab = cc3d.connected_components(m, connectivity=connectivity)
    cnt = cc3d.statistics(lab)["voxel_counts"]
    good = np.where(cnt >= min_vox)[0]
    good = good[good > 0]
    return np.isin(lab, good)


def largest(m, k=1):
    lab = cc3d.connected_components(m, connectivity=6)
    cnt = cc3d.statistics(lab)["voxel_counts"]
    cnt[0] = 0
    ids = np.argsort(cnt)[::-1][:k]
    return np.isin(lab, ids[cnt[ids] > 0])


# ---------------- compact storage ----------------
def save(path, arr):
    """Save compressed .npz with the smallest dtype (bool masks / small labels compress ~100x)."""
    path = Path(path).with_suffix(".npz")
    a = np.asarray(arr)
    if a.dtype.kind in "iu" and a.size and a.min() >= 0:
        a = a.astype(np.uint8 if a.max() < 256 else np.uint16 if a.max() < 65536 else np.uint32)
    np.savez_compressed(path, a=a)
    old = path.with_suffix(".npy")
    if old.exists():
        old.unlink()


def load(path):
    path = Path(path)
    z = path.with_suffix(".npz")
    if z.exists():
        return np.load(z)["a"]
    return np.load(path.with_suffix(".npy"))
