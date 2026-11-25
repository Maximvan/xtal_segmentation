"""08 - Self-supervised denoising (Noise2Void-style blind spot) of the 10 nm crop.

2.5D U-Net: input 5 neighbouring slices, predicts the centre slice. During training ~0.5% of centre-slice
pixels are replaced by a random neighbour and the loss is computed only there, so the network cannot copy
the noisy pixel and learns the clean signal (pixel noise is independent; structure is not).
usage: CUDA_VISIBLE_DEVICES=5 python 08_denoise_n2v.py run0910 [iters]
Outputs: 02_Processed/<run>/den10.npz (uint8, crop10 grid) ; 03_Output/<run>/explore/denoise_*.png
"""
import sys, json, time
import numpy as np, tifffile, torch
import torch.nn as nn, torch.nn.functional as F
from PIL import Image
from xtal_lib import save, RUNS, proc_dir, out_dir

run = sys.argv[1]
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
D, O = proc_dir(run), out_dir(run)
E = O / "explore"; E.mkdir(exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
torch.manual_seed(0); np.random.seed(0)


def load_crop(D, run):
    c = json.load(open(D / "crop10.json"))
    lo, n = c["lo10"], c["shape10"]
    norm = json.load(open(D / "norm.json"))
    resin, gain = np.array(norm["resin"]), norm["gain"]
    raw = tifffile.memmap(RUNS[run]["bin2"])
    v = np.empty(n, np.uint8)
    for z in range(n[0]):
        a = raw[lo[0] + z, lo[1]:lo[1] + n[1], lo[2]:lo[2] + n[2]].astype(np.float32)
        a = (a - resin[min((lo[0] + z) // 2, len(resin) - 1)]) * gain + 200
        v[z, :a.shape[0], :a.shape[1]] = np.clip(a, 0, 255)
    return v, lo, n


class UNet(nn.Module):
    def __init__(s, cin=5, cout=1, b=32):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                                            nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())
        s.e1, s.e2, s.e3, s.e4 = blk(cin, b), blk(b, 2 * b), blk(2 * b, 4 * b), blk(4 * b, 8 * b)
        s.u3, s.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), blk(8 * b, 4 * b)
        s.u2, s.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), blk(4 * b, 2 * b)
        s.u1, s.d1 = nn.ConvTranspose2d(2 * b, b, 2, 2), blk(2 * b, b)
        s.out = nn.Conv2d(b, cout, 1)

    def forward(s, x):
        e1 = s.e1(x); e2 = s.e2(F.max_pool2d(e1, 2)); e3 = s.e3(F.max_pool2d(e2, 2)); e4 = s.e4(F.max_pool2d(e3, 2))
        d3 = s.d3(torch.cat([s.u3(e4), e3], 1)); d2 = s.d2(torch.cat([s.u2(d3), e2], 1)); d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
        return s.out(d1)


if __name__ == "__main__":
    v, lo, n = load_crop(D, run)
    log("crop", v.shape)
    vf = torch.from_numpy(v)  # uint8 on CPU
    dev = "cuda"
    net = UNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=ITERS, pct_start=0.05)
    P, B = 128, 16
    Z, H, W = v.shape
    # sample patches mostly where there is tissue (not plain resin): grey < 192 after light blur proxy
    tis = np.argwhere(v[::4, ::8, ::8] < 190) * [4, 8, 8]
    for it in range(ITERS):
        idx = tis[np.random.randint(len(tis), size=B)]
        xs = []
        for z, y, x in idx:
            z = int(np.clip(z, 2, Z - 3)); y = int(np.clip(y - P // 2, 0, H - P)); x = int(np.clip(x - P // 2, 0, W - P))
            xs.append(vf[z - 2:z + 3, y:y + P, x:x + P])
        x = torch.stack(xs).to(dev).float() / 255.
        tgt = x[:, 2:3].clone()
        m = torch.rand(B, 1, P, P, device=dev) < 0.005
        # blind spot: replace masked centre pixels by a random neighbour (offset in -2..2, not 0,0)
        oy = torch.randint(-2, 3, (B, 1, P, P), device=dev); ox = torch.randint(-2, 3, (B, 1, P, P), device=dev)
        oy[(oy == 0) & (ox == 0)] = 1
        yy = (torch.arange(P, device=dev).view(1, 1, P, 1) + oy).clamp(0, P - 1)
        xx = (torch.arange(P, device=dev).view(1, 1, 1, P) + ox).clamp(0, P - 1)
        bi = torch.arange(B, device=dev).view(B, 1, 1, 1).expand(B, 1, P, P)
        nb = tgt[bi[:, 0], 0, yy[:, 0], xx[:, 0]].unsqueeze(1)
        x[:, 2:3] = torch.where(m, nb, x[:, 2:3])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = net(x).float()
        loss = ((y - tgt) ** 2)[m].mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 1000 == 0:
            log("it", it, "loss", round(loss.item() * 1e3, 4))
    torch.save(net.state_dict(), D / "n2v_unet.pt")

    # inference, slice by slice (pad to multiple of 8)
    net.eval()
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    den = np.empty_like(v)
    with torch.no_grad():
        for z in range(Z):
            zi = np.clip(np.arange(z - 2, z + 3), 0, Z - 1)
            x = torch.from_numpy(v[zi]).to(dev).float()[None] / 255.
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = net(x).float()[0, 0, :H, :W]
            den[z] = (y.clamp(0, 1) * 255).round().byte().cpu().numpy()
    save(D / "den10", den)
    log("saved den10")
    for z in (Z // 4, Z // 2, 3 * Z // 4):
        Image.fromarray(np.vstack([v[z], den[z]])).save(E / f"denoise_z{z + lo[0]:04d}.png")
    y = H // 2
    Image.fromarray(np.vstack([v[:, y, :], den[:, y, :]])).save(E / f"denoise_xz_y{y + lo[1]}.png")
    log("done")
