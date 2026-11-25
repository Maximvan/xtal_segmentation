"""17 - Crystal / sac U-Net at NATIVE resolution (full-res FIB-SEM), trained on Jasmijn's ilastik labels.

Jasmijn's labels live on the bin2 crops (XY binned 2x, Z native). They are mapped onto the full-resolution
stacks (offsets: 02_Processed/sac_unet/crop_offsets_native.json; XY x2, nearest) and the network is trained
on full-res images with a random target pixel size 5-8 nm (covers run0910/0110 5 nm XY, cM 6 nm, 6M 8 nm).
Tests (same as 13): students' labels on run0910 + the user's traced lines, predicted at full res (5 nm XY)
and compared on the 10 nm grid of those labels.
usage: CUDA_VISIBLE_DEVICES=3 python 17_native_sac.py train [iters] | eval
Outputs: 02_Processed/sac_unet_native/{train_*.npz, net.pt}; 03_Output/sac_unet_native/eval.json + pngs
"""
import sys, json, time
import numpy as np, tifffile, torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from xtal_lib import ROOT
import importlib
m13 = importlib.import_module("13_sac_unet")
UNet = m13.UNet

W = ROOT / "02_Processed/sac_unet_native"; W.mkdir(parents=True, exist_ok=True)
O = ROOT / "03_Output/sac_unet_native"; O.mkdir(parents=True, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
RAWD = ROOT / "01_Input/FibSEM xtals"
FULL = {"largeXtal_cM": (RAWD / "EM447_largeXtal_cM_1205.tif", 6.0, 6.0),
        "smallXtal_6M": (RAWD / "EM447_CLEM2_smallXtal_6M_0408/EM447_CLEM2_SmallX_6m.tif", 8.0, 8.0),
        "run0910": (RAWD / "2026_10_01_JasmijnB_FIBSEM_CLEM1/run_0910/EM447_CLEM1_0910.tif", 5.0, 10.0),
        "run0110": (RAWD / "2026_10_01_JasmijnB_FIBSEM_CLEM1/run_0110/EM_447_CLEM1_1001.tif", 5.0, 10.0)}
NC = 3


def neigh(zn):  # neighbour-slice step ~10 nm
    return max(1, int(round(10.0 / zn)))


def prepare():
    off = json.load(open(ROOT / "02_Processed/sac_unet/crop_offsets_native.json"))
    for name, (ilp, _, _, _) in m13.TRAIN.items():
        full, xy, zn = FULL[name]
        o = off[name]
        st = tifffile.memmap(full)
        labs = m13.ilp_labels(ilp)
        dz = neigh(zn)
        # crop extent in full res
        with tifffile.TiffFile(m13.TRAIN[name][1]) as t:
            ch, cw = t.pages[0].shape
        y0f, x0f = 2 * o["y0_bin2"], 2 * o["x0_bin2"]
        Ys = []
        for z in sorted(labs):
            Y = np.full((ch, cw), 255, np.uint8)
            for y0, y1, x0, x1, d in labs[z]:
                sub = Y[y0:y1, x0:x1]; m = d > 0; sub[m] = d[m] - 1
            Ys.append((z, np.repeat(np.repeat(Y, 2, 0), 2, 1)))
        # normalisation on the full-res crop region
        resin = np.zeros(st.shape[0], np.float32)
        for z in range(st.shape[0]):
            a = np.asarray(st[z, y0f:y0f + 2 * ch:8, x0f:x0f + 2 * cw:8]).ravel(); a = a[(a > 2) & (a < 253)]
            resin[z] = np.bincount(a.astype(np.int64), minlength=256).argmax() if a.size else 200
        resin = ndi.median_filter(resin, 9)
        diffs = []
        X, Yo = [], []
        for z, Y in Ys:
            zf = z + o["z_offset"]
            a = np.asarray(st[zf, y0f:y0f + Y.shape[0], x0f:x0f + Y.shape[1]]).astype(np.float32)
            Yc = Y[:a.shape[0], :a.shape[1]]
            if (Yc == 1).any():
                diffs.append(resin[zf] - np.median(a[Yc == 1]))
        gain = 70.0 / max(float(np.median(diffs)), 10)
        for z, Y in Ys:
            zf = z + o["z_offset"]
            zs = np.clip([zf - dz, zf, zf + dz], 0, st.shape[0] - 1)
            a = np.stack([np.asarray(st[k, y0f:y0f + Y.shape[0], x0f:x0f + Y.shape[1]]) for k in zs]).astype(np.float32)
            a = np.clip((a - resin[zs][:, None, None]) * gain + 200, 0, 255).astype(np.uint8)
            X.append(a); Yo.append(Y[:a.shape[1], :a.shape[2]])
        np.savez_compressed(W / f"train_{name}.npz", X=np.stack(X), Y=np.stack(Yo), nm=xy, gain=gain)
        log(name, "slices", len(X), X[0].shape, "nm", xy, "gain", round(gain, 3))


def train(iters):
    D = [np.load(W / f"train_{n}.npz") for n in m13.TRAIN]
    Xs = [torch.from_numpy(d["X"]) for d in D]; Ys = [torch.from_numpy(d["Y"]) for d in D]
    nms = [float(d["nm"]) for d in D]
    cents = [(di, torch.nonzero(Y == k)) for di, Y in enumerate(Ys) for k in range(NC)]
    dev = "cuda"
    net = UNet(cin=3, cout=NC, b=32).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=iters, pct_start=0.05)
    P, B = 384, 8
    rng = np.random.default_rng(0)
    for it in range(iters):
        xb, yb = [], []
        while len(xb) < B:
            di, idx = cents[rng.integers(len(cents))]
            if len(idx) == 0:
                continue
            n, cy, cx = idx[rng.integers(len(idx))].tolist()
            X, Y = Xs[di][n], Ys[di][n]
            tnm = rng.uniform(5.0, 8.0)                 # target pixel size
            s = nms[di] / tnm                            # zoom factor
            ps = int(P / s)
            H, Wd = Y.shape
            y0 = int(np.clip(cy - ps // 2 + rng.integers(-ps // 4, ps // 4 + 1), 0, max(H - ps, 0)))
            x0 = int(np.clip(cx - ps // 2 + rng.integers(-ps // 4, ps // 4 + 1), 0, max(Wd - ps, 0)))
            xp = X[:, y0:y0 + ps, x0:x0 + ps].float()[None]
            yp = Y[y0:y0 + ps, x0:x0 + ps].float()[None, None]
            xp = F.pad(xp, (0, ps - xp.shape[-1], 0, ps - xp.shape[-2]), value=200)
            yp = F.pad(yp, (0, ps - yp.shape[-1], 0, ps - yp.shape[-2]), value=255)
            xp = F.interpolate(xp, size=(P, P), mode="bilinear", align_corners=False)[0]
            yp = F.interpolate(yp, size=(P, P), mode="nearest")[0, 0]
            r = int(rng.integers(4))
            xp, yp = torch.rot90(xp, r, (1, 2)), torch.rot90(yp, r, (0, 1))
            if rng.random() < 0.5:
                xp, yp = xp.flip(-1), yp.flip(-1)
            if rng.random() < 0.5:
                xp = xp[[2, 1, 0]]
            xb.append(xp); yb.append(yp)
        x = torch.stack(xb).to(dev) / 255.
        x = (x - 0.78) * (1 + 0.2 * (torch.rand(B, 1, 1, 1, device=dev) - 0.5)) + 0.78 \
            + 0.04 * (torch.rand(B, 1, 1, 1, device=dev) - 0.5)
        x = x + 0.03 * torch.rand(1, device=dev) * torch.randn_like(x)   # full-res data are noisier
        y = torch.stack(yb).to(dev).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x).float()
        loss = F.cross_entropy(out, y, ignore_index=255)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 500 == 0:
            log("it", it, "loss", round(loss.item(), 4))
    torch.save(net.state_dict(), W / "net.pt")


def load_net():
    net = UNet(cin=3, cout=NC, b=32).cuda()
    net.load_state_dict(torch.load(W / "net.pt", map_location="cuda")); net.eval()
    return net


def full_slices(run, zs10, y0=0, y1=None, x0=0, x1=None):
    """run0910/0110: full-res (5 nm XY) 3-slice inputs for 10 nm slice indices / 10 nm crop coords"""
    full, xy, zn = FULL[run]
    st = tifffile.memmap(full)
    n = json.load(open(ROOT / f"02_Processed/{run}/norm.json"))
    res = np.array(n["resin"])
    # full-res intensities: same offset/gain as bin2 (binning averages, levels unchanged)
    Y0, X0 = 2 * y0, 2 * x0
    Y1 = None if y1 is None else 2 * y1
    X1 = None if x1 is None else 2 * x1
    for z in zs10:
        zz = np.clip([z - 1, z, z + 1], 0, st.shape[0] - 1)
        a = np.stack([st[k, Y0:Y1, X0:X1] for k in zz]).astype(np.float32)
        yield z, np.clip((a - res[np.minimum(zz // 2, len(res) - 1)][:, None, None]) * n["gain"] + 200, 0, 255)


_PRED_FULL = m13.predict  # original full-image predictor (kept before any patching)


def predict10(net, X):
    """predict at full res, return class probs averaged 2x2 onto the 10 nm grid (odd sizes: edge-padded)"""
    p = _PRED_FULL(net, X)  # (NC, H, W)
    p = np.pad(p, ((0, 0), (0, p.shape[1] % 2), (0, p.shape[2] % 2)), mode="edge")
    return p.reshape(NC, p.shape[1] // 2, 2, p.shape[2] // 2, 2).mean((2, 4))


def evaluate():
    net = load_net()
    m13.load_net = lambda: net
    # reuse 13's evaluation with full-res inputs and 2x-downsampled predictions
    m13.run_slices = full_slices
    m13.predict = predict10
    m13.O = O
    m13.evaluate()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        prepare()
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
        evaluate()
    elif mode == "eval":
        evaluate()
