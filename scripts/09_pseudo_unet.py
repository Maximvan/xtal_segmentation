"""09 - 2.5D U-Net trained on our own 10 nm labels (pseudo-labels), on the denoised crop.

Classes: 0 resin, 1 crystal, 2 cytoplasm, 3 nucleus, 4 mitochondrion, 5 crystal vacuole, 6 granule/dense,
7 other tissue. A +-20 nm band around every label boundary is ignored (uncertain), so the network learns
from confident interiors and decides the borders itself from texture. Compare with the rule-based labels;
later this network can be fine-tuned on Jasmijn's annotations.
usage: CUDA_VISIBLE_DEVICES=5 python 09_pseudo_unet.py run0910 [iters]
Outputs: 02_Processed/<run>/pred10.npz (uint8 classes), unet_pseudo.pt ;
         03_Output/<run>/explore/unet_*.png, unet_agreement.json
"""
import sys, json, time
import numpy as np, torch, edt
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
from xtal_lib import save, load, proc_dir, out_dir, NT
import importlib
n2v = importlib.import_module("08_denoise_n2v")

run = sys.argv[1]
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
D, O = proc_dir(run), out_dir(run)
E = O / "explore"; E.mkdir(exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
torch.manual_seed(0); np.random.seed(0)
NC = 8
NAMES = ["resin", "crystal", "cytoplasm", "nucleus", "mito", "vacuole", "granule/dense", "other tissue"]

# image
if (D / "den10.npz").exists():
    v = load(D / "den10"); src = "denoised"
else:
    v, _, _ = n2v.load_crop(D, run); src = "raw"
log("image", src, v.shape)
# pseudo labels
L = load(D / "lab10")
Y = np.zeros(L.shape, np.uint8)
Y[L == 3] = 7
Y[(L >= 10) & (L < 30)] = 2
Y[L >= 30] = 3
Y[L == 2] = 1
Y[(load(D / "mito10") > 0)] = 4
Y[load(D / "vac10").astype(bool) & (L != 2)] = 5
g = load(D / "gran10").astype(bool)
if (D / "dense10.npz").exists():
    g |= load(D / "dense10").astype(bool)
Y[g] = 6
del L, g
# ignore band (+-2 px) around class boundaries, per slice chunk to save memory
for z0 in range(0, Y.shape[0], 64):
    sl = slice(max(z0 - 2, 0), min(z0 + 66, Y.shape[0]))
    b = find_boundaries(Y[sl], mode="thick")
    band = edt.edt(~b, parallel=NT) <= 1.5
    k0 = z0 - sl.start
    Yc = Y[sl].copy(); Yc[band] = 255
    Y[z0:z0 + 64] = Yc[k0:k0 + 64]
cnt = np.bincount(Y[Y < 255].ravel(), minlength=NC).astype(float)
log("class voxels", dict(zip(NAMES, cnt.astype(int).tolist())))
wts = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float32)
wts = (wts / wts.mean()).clamp(0.2, 20)

dev = "cuda"
net = n2v.UNet(cin=5, cout=NC, b=32).to(dev)
opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=ITERS, pct_start=0.05)
Z, H, W = v.shape
P, B = 192, 8
vt, yt = torch.from_numpy(v), torch.from_numpy(Y)
# patch centres: half on tissue (non-resin labels), half anywhere
fg = np.argwhere(Y[::4, ::8, ::8] % 255 > 0) * [4, 8, 8]
for it in range(ITERS):
    xs, ys = [], []
    for j in range(B):
        if j % 2 == 0:
            z, y, x = fg[np.random.randint(len(fg))]
        else:
            z, y, x = np.random.randint(Z), np.random.randint(H), np.random.randint(W)
        z = int(np.clip(z, 2, Z - 3)); y = int(np.clip(y - P // 2, 0, H - P)); x = int(np.clip(x - P // 2, 0, W - P))
        a, l = vt[z - 2:z + 3, y:y + P, x:x + P], yt[z, y:y + P, x:x + P]
        if np.random.rand() < 0.5: a, l = a.flip(-1), l.flip(-1)
        if np.random.rand() < 0.5: a, l = a.flip(-2), l.flip(-2)
        xs.append(a); ys.append(l)
    x = torch.stack(xs).to(dev).float() / 255.
    x = x * (1 + 0.1 * (torch.rand(B, 1, 1, 1, device=dev) - 0.5)) + 0.05 * (torch.rand(B, 1, 1, 1, device=dev) - 0.5)
    y = torch.stack(ys).to(dev).long()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = net(x).float()
    loss = F.cross_entropy(out, y, weight=wts.to(dev), ignore_index=255)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 1000 == 0:
        log("it", it, "loss", round(loss.item(), 4))
torch.save(net.state_dict(), D / "unet_pseudo.pt")

net.eval()
ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
pred = np.empty(v.shape, np.uint8)
with torch.no_grad():
    for z in range(Z):
        zi = np.clip(np.arange(z - 2, z + 3), 0, Z - 1)
        x = F.pad(torch.from_numpy(v[zi]).to(dev).float()[None] / 255., (0, pw, 0, ph), mode="reflect")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred[z] = net(x).float()[0, :, :H, :W].argmax(0).byte().cpu().numpy()
pred = ndi.median_filter(pred, size=(3, 1, 1))  # light Z consistency
save(D / "pred10", pred)

# agreement with the rule-based labels on confident voxels
m = Y < 255
agree = {NAMES[k]: dict(rule_vox=int((Y[m] == k).sum()),
                        recall=float(((pred[m] == k) & (Y[m] == k)).sum() / max((Y[m] == k).sum(), 1)),
                        precision=float(((pred[m] == k) & (Y[m] == k)).sum() / max((pred[m] == k).sum(), 1)))
         for k in range(NC)}
json.dump(dict(source=src, iters=ITERS, agreement=agree), open(E / "unet_agreement.json", "w"), indent=1)
log(agree)
COL = np.array([(200, 200, 200), (86, 180, 233), (230, 159, 0), (0, 114, 178), (213, 94, 0), (240, 228, 66),
                (0, 158, 115), (150, 150, 150)], np.uint8)
for z in (Z // 4, Z // 2, 3 * Z // 4):
    im = np.stack([v[z]] * 3, -1)
    def ov(lab):
        o = im.copy(); b = find_boundaries(lab, mode="inner")
        o[b] = COL[np.minimum(lab[b], NC - 1)]
        return o
    yl = Y[z].copy(); yl[yl == 255] = 0
    Image.fromarray(np.vstack([ov(yl), ov(pred[z])])).save(E / f"unet_z{z:04d}.png")
log("done")
