"""03 - Mitochondria probability with pretrained MitoNet_v1 (empanada, Conrad & Narayan 2023).

Runs the torchscript model slice-by-slice (XY) on the bin2 stack (10 nm XY, 10 nm Z),
keeps the semantic sigmoid, averages Z pairs and downsamples XY 2x to match img20 (20 nm iso).
Output: 02_Processed/<run>/mito_prob20.npz and mito_prob10.npz (bin2 grid) (uint8, prob*255)
"""
from pathlib import Path
import numpy as np, tifffile, torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from xtal_lib import save, load, RUNS, proc_dir
RUN = sys.argv[1]
SRC = RUNS[RUN]["bin2"]
OUT = proc_dir(RUN)
dev = "cuda"
m = torch.jit.load(str(ROOT / "env/models/MitoNet_v1.pth"), map_location=dev).eval()

stack = tifffile.memmap(SRC)  # large ImageJ files are contiguous: pages[] is unreliable
if True:
    n, H, W = stack.shape
    Z = n // 2
    out = np.zeros((Z, H // 2, W // 2), np.uint8)
    out10 = np.zeros((n, H, W), np.uint8)
    ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
    acc = None
    for z in range(Z * 2):
        s = np.asarray(stack[z]).astype(np.float32)
        # match training intensity range: per-slice robust scaling of the resin peak, like 01_preprocess
        valid = (s > 0.5) & (s < 254.5)
        vals = s[valid]
        hist, e = np.histogram(vals, 256, (0, 256))
        resin = e[np.argmax(hist)] + .5
        lo = np.percentile(vals, 1)
        s = (s - resin) * (140 / max(resin - lo, 1)) + 200
        s[~valid] = 200
        x = np.pad(np.clip(s, 0, 255) / 255., ((0, ph), (0, pw)), mode="reflect")
        x = torch.from_numpy(((x - 0.57571) / 0.12765).astype(np.float32))[None, None].to(dev)
        with torch.no_grad():
            p = torch.sigmoid(m(x)["sem_logits"].float())[:, :, :H, :W]
        out10[z] = np.clip(p[0, 0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        p = F.avg_pool2d(p, 2)[0, 0].cpu().numpy()
        if z % 2 == 0:
            acc = p
        else:
            out[z // 2] = np.clip((acc + p) / 2 * 255, 0, 255).astype(np.uint8)
        if z % 100 == 0:
            print(z, flush=True)

save(OUT / "mito_prob20", out)
save(OUT / "mito_prob10", out10)
print("done", out.shape)
