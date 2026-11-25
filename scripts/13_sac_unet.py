"""13 - Crystal / sac (vacuole) U-Net trained on Jasmijn's ilastik brush strokes; tested on independent labels.

Train : LargeXtal_CM_test.ilp (large crystal cM_1205, 12 nm) + Small xtal.ilp (small crystal 6M, 16 nm),
        Jasmijn's own sparse labels (Background / Crystal / Sac). Loss only on painted pixels.
Test 1: students' labels on run0910 (Large xtal_3.ilp) - other sample, other annotators, never trained on.
Test 2: the user's traced lines (03_Output/annotate/user_lines.json): crystal + outer sac membrane.
All data: per-slice resin offset + one gain per dataset (resin -> 200, crystal -> 130), resampled to 10 nm.
Input: 3 neighbouring slices ~10 nm apart; 2D U-Net (shared with 08).
usage: CUDA_VISIBLE_DEVICES=5 python 13_sac_unet.py train [iters]   |   ... eval
Outputs: 02_Processed/sac_unet/{train_*.npz, sac_unet.pt}, 03_Output/sac_unet/{eval.json, *.png}
"""
import sys, json, time
import numpy as np, tifffile, h5py, torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from xtal_lib import ROOT, RUNS
import importlib
UNet = importlib.import_module("08_denoise_n2v").UNet

ILA = ROOT / "01_Input/FibSEM xtals/Ilastik data/JasmijnB_UBla"
W = ROOT / "02_Processed/sac_unet"; W.mkdir(parents=True, exist_ok=True)
O = ROOT / "03_Output/sac_unet"; O.mkdir(parents=True, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
TARGET = 10.0  # nm
# name: (ilp, raw tif, xy nm, z nm)
TRAIN = {"largeXtal_cM": (ILA / "LargeXtal_CM_test.ilp", ILA / "EM447_largeXtal_cM_1205_bin2_crop.tif", 12.0, 6.0),
         "smallXtal_6M": (ILA / "Small xtal/Small xtal.ilp", ILA / "Small xtal/EM_447_CLEM2_SmallX_6M_bin2_crop.tif", 16.0, 8.0)}
NC = 3  # 0 background, 1 crystal, 2 sac   (ilastik labels 1,2,3 -> 0,1,2; 255 = unlabelled)


def ilp_labels(ilp):
    """sparse labels {z: (y0,y1,x0,x1,array)} from an ilastik project (z,y,x,c blocks)"""
    out = {}
    with h5py.File(ilp, "r") as h:
        for b in h["PixelClassification/LabelSets/labels000"].values():
            sl = b.attrs["blockSlice"]; sl = sl.decode() if isinstance(sl, bytes) else sl
            (z0, z1), (y0, y1), (x0, x1) = [tuple(map(int, p.split(":"))) for p in sl.strip("[]").split(",")[:3]]
            d = b[()][..., 0]
            for k, z in enumerate(range(z0, z1)):
                if d[k].any():
                    out.setdefault(z, []).append((y0, y1, x0, x1, d[k]))
    return out


def dataset_norm(stack, lab=None):
    """per-slice resin offset (mode) + one gain (resin-crystal -> 70). lab: (zs, Y) to get the crystal level."""
    Z = stack.shape[0]
    resin = np.zeros(Z, np.float32)
    for z in range(Z):
        a = np.asarray(stack[z][::4, ::4]).ravel(); a = a[(a > 2) & (a < 253)]
        resin[z] = np.bincount(a.astype(np.int64), minlength=256).argmax() if a.size else 200
    resin = ndi.median_filter(resin, 9)
    diffs = []
    if lab is not None:
        for z, Y in lab:
            a = np.asarray(stack[z]).astype(np.float32)
            diffs.append(resin[z] - np.median(a[Y == 1])) if (Y == 1).any() else None
    diff = float(np.median([d for d in diffs if d is not None])) if diffs else 97.0
    return resin, 70.0 / max(diff, 10)


def norm_slices(stack, zs, resin, gain):
    a = np.stack([np.asarray(stack[z]) for z in zs]).astype(np.float32)
    return np.clip((a - resin[zs][:, None, None]) * gain + 200, 0, 255)


def resize(a, f, order):
    return ndi.zoom(a, (1, f, f) if a.ndim == 3 else (f, f), order=order, prefilter=False)


def prepare():
    for name, (ilp, tif, xy, zn) in TRAIN.items():
        labs = ilp_labels(ilp)
        st = tifffile.memmap(tif)
        dz = max(1, int(round(TARGET / zn)))
        Ys = []
        for z in sorted(labs):
            Y = np.full(st.shape[1:], 255, np.uint8)
            for y0, y1, x0, x1, d in labs[z]:
                sub = Y[y0:y1, x0:x1]; m = d > 0; sub[m] = d[m] - 1
            Ys.append((z, Y))
        resin, gain = dataset_norm(st, [(z, np.where(Y == 255, 0, Y + 1) - 1) for z, Y in Ys])
        X, Yo = [], []
        f = xy / TARGET
        for z, Y in Ys:
            zs = np.clip([z - dz, z, z + dz], 0, st.shape[0] - 1)
            X.append(resize(norm_slices(st, zs, resin, gain), f, 1).astype(np.uint8))
            Yo.append(resize(Y, f, 0))
        np.savez_compressed(W / f"train_{name}.npz", X=np.stack(X), Y=np.stack(Yo), zs=np.array([z for z, _ in Ys]),
                            gain=gain)
        log(name, "slices", len(X), "shape", X[0].shape, "gain", round(gain, 3),
            "labelled px", {c: int(sum((y == k).sum() for y in Yo)) for k, c in enumerate(["bg", "crystal", "sac"])})


def train(iters):
    D = [np.load(W / f"train_{n}.npz") for n in TRAIN]
    Xs = [torch.from_numpy(d["X"]) for d in D]
    Ys = [torch.from_numpy(d["Y"]) for d in D]
    # patch centres: labelled pixels, balanced per class and dataset
    cents = []
    for di, Y in enumerate(Ys):
        for k in range(NC):
            idx = torch.nonzero(Y == k)
            cents.append((di, k, idx))
    dev = "cuda"
    net = UNet(cin=3, cout=NC, b=32).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=iters, pct_start=0.05)
    P, B = 256, 12
    rng = np.random.default_rng(0)
    for it in range(iters):
        xb, yb = [], []
        for j in range(B):
            di, k, idx = cents[rng.integers(len(cents))]
            if len(idx) == 0:
                continue
            n, cy, cx = idx[rng.integers(len(idx))].tolist()
            X, Y = Xs[di][n], Ys[di][n]
            # scale augmentation 0.8-1.25 around the 10 nm grid
            s = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25))))
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
            r = rng.integers(4)
            xp, yp = torch.rot90(xp, r, (1, 2)), torch.rot90(yp, r, (0, 1))
            if rng.random() < 0.5:
                xp, yp = xp.flip(-1), yp.flip(-1)
            if rng.random() < 0.5:  # swap the two neighbour slices (Z direction is arbitrary)
                xp = xp[[2, 1, 0]]
            xb.append(xp); yb.append(yp)
        x = torch.stack(xb).to(dev) / 255.
        # intensity augmentation: gain/offset/noise jitter (different scanners / sessions)
        x = (x - 0.78) * (1 + 0.2 * (torch.rand(len(xb), 1, 1, 1, device=dev) - 0.5)) + 0.78 \
            + 0.04 * (torch.rand(len(xb), 1, 1, 1, device=dev) - 0.5)
        x = x + 0.02 * torch.rand(1, device=dev) * torch.randn_like(x)
        y = torch.stack(yb).to(dev).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x).float()
        loss = F.cross_entropy(out, y, ignore_index=255)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 500 == 0:
            log("it", it, "loss", round(loss.item(), 4))
    torch.save(net.state_dict(), W / "sac_unet.pt")


def load_net():
    net = UNet(cin=3, cout=NC, b=32).cuda()
    net.load_state_dict(torch.load(W / "sac_unet.pt", map_location="cuda")); net.eval()
    return net


def predict(net, X):
    """X: (3,H,W) normalised 0-255 at 10 nm -> class probs (NC,H,W)"""
    H, Wd = X.shape[1:]
    ph, pw = (8 - H % 8) % 8, (8 - Wd % 8) % 8
    with torch.no_grad():
        x = F.pad(torch.from_numpy(np.ascontiguousarray(X)).float().cuda()[None] / 255., (0, pw, 0, ph), mode="reflect")
        p = 0
        for r in range(4):  # rotation test-time augmentation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(torch.rot90(x, r, (2, 3))).float()
            p = p + torch.softmax(torch.rot90(o, -r, (2, 3)), 1)
        return (p / 4)[0, :, :H, :Wd].cpu().numpy()


def run_slices(run, zs, y0=0, y1=None, x0=0, x1=None):
    """normalised 3-slice inputs from our runs (bin2 = 10 nm), using our 01 normalisation"""
    raw = tifffile.memmap(RUNS[run]["bin2"])
    n = json.load(open(ROOT / f"02_Processed/{run}/norm.json"))
    res = np.array(n["resin"])
    for z in zs:
        zz = np.clip([z - 1, z, z + 1], 0, raw.shape[0] - 1)
        a = np.stack([raw[k, y0:y1, x0:x1] for k in zz]).astype(np.float32)
        yield z, np.clip((a - res[np.minimum(zz // 2, len(res) - 1)][:, None, None]) * n["gain"] + 200, 0, 255)


def evaluate():
    net = load_net()
    res = {}
    # Test 1: students' labels on run0910
    labs = ilp_labels(ILA / "Large xtal/Large xtal_3.ilp")
    raw_shape = tifffile.memmap(RUNS["run0910"]["bin2"]).shape
    zs = sorted(labs)
    conf = np.zeros((NC, NC), np.int64)  # rows: student label, cols: prediction
    for z, X in run_slices("run0910", zs):
        pr = predict(net, X).argmax(0)
        pr = np.pad(pr, ((0, max(0, raw_shape[1] - pr.shape[0])), (0, max(0, raw_shape[2] - pr.shape[1]))), mode="edge")
        for y0, y1, x0, x1, d in labs[z]:
            m = d > 0
            np.add.at(conf, (d[m].astype(int) - 1, pr[y0:y1, x0:x1][m]), 1)
    names = ["background", "crystal", "sac"]
    t1 = {names[k]: dict(student_px=int(conf[k].sum()),
                         recall=float(conf[k, k] / max(conf[k].sum(), 1)),
                         precision=float(conf[k, k] / max(conf[:, k].sum(), 1)),
                         iou=float(conf[k, k] / max(conf[k].sum() + conf[:, k].sum() - conf[k, k], 1))) for k in range(NC)}
    t1["confusion_rows_student_cols_pred"] = conf.tolist()
    res["students_run0910"] = t1
    log("students run0910", json.dumps({k: {a: round(b, 3) for a, b in v.items() if a != "student_px"} for k, v in t1.items() if k in names}))
    # Test 2: user's traced lines
    A = ROOT / "03_Output/annotate"
    meta = json.load(open(A / "crops.json")); ann = json.load(open(A / "user_lines.json"))
    t2 = {}
    tiles = []
    for name, a in ann.items():
        if not a or "vacuole" not in a and "crystal" not in a:
            continue
        m = meta[name]
        h, w = np.array(Image.open(A / name)).shape
        _, X = next(run_slices(m["run"], [m["z10"]], m["y0"], m["y0"] + h, m["x0"], m["x0"] + w))
        pr = predict(net, X).argmax(0)[:h, :w]
        cr = ndi.binary_fill_holes(ndi.binary_opening(pr == 1, iterations=2))
        lab_, nl = ndi.label(cr)
        if nl:
            sz = ndi.sum(cr, lab_, range(1, nl + 1)); cr = np.isin(lab_, 1 + np.where(sz > 2000)[0])
        sac = (pr == 2) & ~cr
        lab_, nl = ndi.label(sac | cr)
        keep = np.unique(lab_[cr]); sac = np.isin(lab_, keep[keep > 0]) & ~cr
        out = {}
        from skimage.segmentation import find_boundaries
        for k, mask in (("crystal", cr), ("vacuole", sac | cr)):
            if k not in a:
                continue
            b = find_boundaries(mask, mode="inner" if k == "crystal" else "outer")
            if k == "vacuole":
                b &= ~cr
            if not b.any():
                out[k] = "none predicted"; continue
            d = ndi.distance_transform_edt(~b)
            p = np.array(a[k]); dd = d[p[:, 0].clip(0, h - 1), p[:, 1].clip(0, w - 1)] * 10
            out[k] = dict(median_nm=float(np.median(dd)), within30=float(np.mean(dd <= 30)))
        t2[name] = out
        log(name, out)
        k_ = max(1, X.shape[1] // pr.shape[0])  # full-res input (17): show on the prediction grid
        o = np.stack([np.clip((X[1, ::k_, ::k_][:h, :w] - 110) / 105 * 255, 0, 255)] * 3, -1).astype(np.uint8)
        o[sac] = (0.5 * o[sac] + 0.5 * np.array((240, 228, 66))).astype(np.uint8)
        o[find_boundaries(cr)] = (0, 0, 160)
        for k, col in (("vacuole", (0, 0, 0)), ("crystal", (213, 94, 0))):
            if k in a:
                p = np.array(a[k]); o[p[:, 0].clip(0, h - 1), p[:, 1].clip(0, w - 1)] = col
        Image.fromarray(o).save(O / f"test_{name}")
    res["user_lines"] = t2
    json.dump(res, open(O / "eval.json", "w"), indent=1)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ("train", "all"):
        prepare()
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 15000)
    if mode in ("eval", "all", "train"):
        evaluate()
