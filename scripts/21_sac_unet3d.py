"""21 - 3D crystal / sac U-Net (16-slice context) trained on Jasmijn's ilastik labels; same tests as 13.

Jasmijn's labels are sparse 2D strokes on single slices. For every labelled slice a 16-slice stack around it is
resampled to the 10 nm grid (Z: linear interpolation at 10 nm steps; XY as in 13), and the loss is computed on
the labelled slice only (index 8 of 16) - the network still uses 3D context, so its prediction is consistent
between slices (the 2D network made the sac fragmented).
usage: CUDA_VISIBLE_DEVICES=3 python 21_sac_unet3d.py train [iters] | eval
Outputs: 02_Processed/sac_unet3d/{train_*.npz, net3d.pt}, 03_Output/sac_unet3d/eval.json + test pngs
"""
import sys, json, time
import numpy as np, tifffile, torch
import torch.nn as nn, torch.nn.functional as F
from scipy import ndimage as ndi
from xtal_lib import ROOT, RUNS
import importlib
m13 = importlib.import_module("13_sac_unet")

W = ROOT / "02_Processed/sac_unet3d"; W.mkdir(parents=True, exist_ok=True)
O = ROOT / "03_Output/sac_unet3d"; O.mkdir(parents=True, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
D, C = 16, 8  # slices per window, index of the labelled slice
NC = 3


class UNet3D(nn.Module):
    def __init__(s, b=16):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU(),
                                            nn.Conv3d(o, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU())
        s.e1, s.e2, s.e3, s.e4 = blk(1, b), blk(b, 2 * b), blk(2 * b, 4 * b), blk(4 * b, 8 * b)
        s.p1, s.p2, s.p3 = nn.MaxPool3d((1, 2, 2)), nn.MaxPool3d(2), nn.MaxPool3d(2)
        s.u3, s.d3 = nn.ConvTranspose3d(8 * b, 4 * b, 2, 2), blk(8 * b, 4 * b)
        s.u2, s.d2 = nn.ConvTranspose3d(4 * b, 2 * b, 2, 2), blk(4 * b, 2 * b)
        s.u1, s.d1 = nn.ConvTranspose3d(2 * b, b, (1, 2, 2), (1, 2, 2)), blk(2 * b, b)
        s.out = nn.Conv3d(b, NC, 1)

    def forward(s, x):
        e1 = s.e1(x); e2 = s.e2(s.p1(e1)); e3 = s.e3(s.p2(e2)); e4 = s.e4(s.p3(e3))
        d3 = s.d3(torch.cat([s.u3(e4), e3], 1)); d2 = s.d2(torch.cat([s.u2(d3), e2], 1))
        d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
        return s.out(d1)


def zstack(st, z, zn, resin, gain, f):
    """16 slices at 10 nm steps centred on slice z (source Z step zn nm), normalised, XY scaled by f"""
    pos = z + (np.arange(D) - C) * (10.0 / zn)
    out = []
    for p in pos:
        p = float(np.clip(p, 0, st.shape[0] - 1)); i0 = int(np.floor(p)); i1 = min(i0 + 1, st.shape[0] - 1); w = p - i0
        a = (1 - w) * np.asarray(st[i0], np.float32) + w * np.asarray(st[i1], np.float32)
        r = (1 - w) * resin[i0] + w * resin[i1]
        out.append(np.clip((a - r) * gain + 200, 0, 255))
    return m13.resize(np.stack(out), f, 1).astype(np.uint8)


def prepare():
    for name, (ilp, tif, xy, zn) in m13.TRAIN.items():
        labs = m13.ilp_labels(ilp)
        st = tifffile.memmap(tif)
        Ys = []
        for z in sorted(labs):
            Y = np.full(st.shape[1:], 255, np.uint8)
            for y0, y1, x0, x1, d in labs[z]:
                sub = Y[y0:y1, x0:x1]; m = d > 0; sub[m] = d[m] - 1
            Ys.append((z, Y))
        resin, gain = m13.dataset_norm(st, [(z, np.where(Y == 255, 0, Y + 1) - 1) for z, Y in Ys])
        f = xy / 10.0
        X = np.stack([zstack(st, z, zn, resin, gain, f) for z, _ in Ys])
        Yo = np.stack([m13.resize(Y, f, 0) for _, Y in Ys])
        np.savez_compressed(W / f"train_{name}.npz", X=X, Y=Yo)
        log(name, "stacks", X.shape)


def train(iters):
    Ds = [np.load(W / f"train_{n}.npz") for n in m13.TRAIN]
    Xs = [torch.from_numpy(d["X"]) for d in Ds]; Ys = [torch.from_numpy(d["Y"]) for d in Ds]
    cents = [(di, torch.nonzero(Y == k)) for di, Y in enumerate(Ys) for k in range(NC)]
    dev = "cuda"
    net = UNet3D().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=iters, pct_start=0.05)
    P, B = 160, 4
    rng = np.random.default_rng(0)
    for it in range(iters):
        xb, yb = [], []
        while len(xb) < B:
            di, idx = cents[rng.integers(len(cents))]
            if len(idx) == 0:
                continue
            n, cy, cx = idx[rng.integers(len(idx))].tolist()
            X, Y = Xs[di][n], Ys[di][n]
            H, Wd = Y.shape
            y0 = int(np.clip(cy - P // 2 + rng.integers(-P // 4, P // 4 + 1), 0, H - P))
            x0 = int(np.clip(cx - P // 2 + rng.integers(-P // 4, P // 4 + 1), 0, Wd - P))
            xp = X[:, y0:y0 + P, x0:x0 + P].float()
            yp = Y[y0:y0 + P, x0:x0 + P].clone()
            r = int(rng.integers(4))
            xp, yp = torch.rot90(xp, r, (1, 2)), torch.rot90(yp, r, (0, 1))
            if rng.random() < 0.5:
                xp, yp = xp.flip(-1), yp.flip(-1)
            if rng.random() < 0.5:
                xp = xp.flip(0)  # Z direction is arbitrary (labelled slice stays at the centre)
            xb.append(xp[None]); yb.append(yp)
        x = torch.stack(xb).to(dev) / 255.
        x = (x - 0.78) * (1 + 0.2 * (torch.rand(B, 1, 1, 1, 1, device=dev) - 0.5)) + 0.78 \
            + 0.04 * (torch.rand(B, 1, 1, 1, 1, device=dev) - 0.5)
        x = x + 0.02 * torch.rand(1, device=dev) * torch.randn_like(x)
        y = torch.stack(yb).to(dev).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x).float()[:, :, C]
        loss = F.cross_entropy(out, y, ignore_index=255)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 500 == 0:
            log("it", it, "loss", round(loss.item(), 4))
    torch.save(net.state_dict(), W / "net3d.pt")


def load_net():
    net = UNet3D().cuda()
    net.load_state_dict(torch.load(W / "net3d.pt", map_location="cuda")); net.eval()
    return net


def predict_slice(net, X16, tile=768):
    """X16: (16, H, W) normalised at 10 nm -> class probs (NC, H, W) of the centre slice, XY tiles with overlap"""
    Hh, Ww = X16.shape[1:]
    out = np.zeros((NC, Hh, Ww), np.float32); cnt = np.zeros((Hh, Ww), np.float32)
    step = tile - 128
    with torch.no_grad():
        for y0 in range(0, max(Hh - 128, 1), step):
            for x0 in range(0, max(Ww - 128, 1), step):
                y1, x1 = min(y0 + tile, Hh), min(x0 + tile, Ww)
                a = X16[:, y0:y1, x0:x1]
                ph, pw = (-a.shape[1]) % 8, (-a.shape[2]) % 8
                x = F.pad(torch.from_numpy(np.ascontiguousarray(a)).float().cuda()[None, None] / 255., (0, pw, 0, ph, 0, 0), mode="replicate")
                p = 0
                for r in range(4):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o = net(torch.rot90(x, r, (3, 4))).float()[:, :, C]
                    p = p + torch.softmax(torch.rot90(o, -r, (2, 3)), 1)
                p = (p / 4)[0, :, :a.shape[1], :a.shape[2]].cpu().numpy()
                out[:, y0:y1, x0:x1] += p; cnt[y0:y1, x0:x1] += 1
    return out / np.maximum(cnt, 1)


def run_stacks(run, zs, y0=0, y1=None, x0=0, x1=None):
    """16-slice inputs from our runs (bin2 = 10 nm grid, destriped where available) with our 01 normalisation"""
    raw = tifffile.memmap(RUNS[run]["bin2"])
    n = json.load(open(ROOT / f"02_Processed/{run}/norm.json"))
    res = np.array(n["resin"])
    for z in zs:
        zz = np.clip(np.arange(z - C, z - C + D), 0, raw.shape[0] - 1)
        a = np.stack([raw[k, y0:y1, x0:x1] for k in zz]).astype(np.float32)
        yield z, np.clip((a - res[np.minimum(zz // 2, len(res) - 1)][:, None, None]) * n["gain"] + 200, 0, 255)


def evaluate():
    net = load_net()
    m13.load_net = lambda: net
    m13.run_slices = run_stacks
    m13.predict = predict_slice
    m13.O = O
    m13.evaluate()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        prepare()
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 12000)
    if mode in ("train", "eval"):
        evaluate()
