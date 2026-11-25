"""11 - Vacuole / tissue pixel classifier trained on hand-traced lines (ilastik-style random forest).

Labels from the user's lines on 03_Output/annotate crops (user_lines.json; red crystal, yellow outer
vacuole membrane, blue cell membrane, green nucleus): crystal body, vacuole = between crystal and the yellow
line, cytoplasm = just beyond the yellow line (and inside cells away from nuclei), nucleus = inside green,
resin = bright & outside blue. Features (2.5D, centre of a 7-slice stack): Gaussian, local std, gradient,
Hessian eigenvalues at several scales, median, + distance to the crystal.
usage:
  python 11_vacuole_classifier.py eval           leave-one-crop-out boundary accuracy vs the current rule
  python 11_vacuole_classifier.py apply run0910  train on all crops, write vac10 (old kept as vac10_rule)
"""
import sys, json, time
import numpy as np, tifffile, edt, cc3d
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
import torch
NEGATIVE = ["05_run0910_vacuole_z300.png"]  # user: no vacuole visible here (direct contact)


class MLP:
    """small GPU pixel classifier on standardised features (stands in for a random forest; much faster)"""
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.classes_ = np.unique(y)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32, device="cuda")
        yt = torch.tensor(np.searchsorted(self.classes_, y), device="cuda")
        nf, nc = X.shape[1], len(self.classes_)
        self.net = torch.nn.Sequential(torch.nn.Linear(nf, 128), torch.nn.SiLU(), torch.nn.Dropout(0.1),
                                       torch.nn.Linear(128, 128), torch.nn.SiLU(), torch.nn.Linear(128, nc)).cuda()
        opt = torch.optim.AdamW(self.net.parameters(), 2e-3, weight_decay=1e-4)
        for ep in range(3000):
            idx = torch.randint(0, len(yt), (4096,), device="cuda")
            loss = torch.nn.functional.cross_entropy(self.net(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval()
        return self

    def predict_proba(self, X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 1 << 20):
                xt = torch.tensor((X[i:i + (1 << 20)] - self.mu) / self.sd, dtype=torch.float32, device="cuda")
                out.append(torch.softmax(self.net(xt), 1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes_)), np.float32)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def RandomForestClassifier(*a, **k):
    return MLP()
from xtal_lib import RUNS, ROOT, load, save, proc_dir, NT

A = ROOT / "03_Output/annotate"
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
CLS = ["resin", "crystal", "vacuole", "cytoplasm", "nucleus"]
DCAP = 80


def norm_stack(run, z10, y0, y1, x0, x1, half=3):
    raw = tifffile.memmap(RUNS[run]["bin2"])
    n = json.load(open(ROOT / f"02_Processed/{run}/norm.json"))
    zr = np.clip(np.arange(z10 - half, z10 + half + 1), 0, raw.shape[0] - 1)
    a = np.stack([raw[z, y0:y1, x0:x1] for z in zr]).astype(np.float32)
    res = np.array(n["resin"])[np.minimum(zr // 2, len(n["resin"]) - 1)]
    return (a - res[:, None, None]) * n["gain"] + 200


def features(stack, crys):
    """stack: (7, H, W) normalised; crys: (H, W) bool crystal. returns (H, W, F) float32"""
    try:
        import cupy as cp
        from cupyx.scipy import ndimage as nd
        xp = cp
    except Exception:
        nd, xp = ndi, np
    s = xp.asarray(stack)
    F = []
    for sg in (1, 2, 4, 8):
        g = nd.gaussian_filter(s, (1.5, sg, sg))
        F.append(g[3])
        m = nd.gaussian_filter(g, (0, 2 * sg, 2 * sg))
        F.append(xp.sqrt(xp.maximum(nd.gaussian_filter(g * g, (0, 2 * sg, 2 * sg)) - m * m, 0))[3])
        gy, gx = nd.sobel(g[3], 0), nd.sobel(g[3], 1)
        F.append(xp.sqrt(gy * gy + gx * gx))
    for sg in (1.5, 3, 6):
        g = nd.gaussian_filter(s[3], sg)
        hyy = nd.sobel(nd.sobel(g, 0), 0); hxx = nd.sobel(nd.sobel(g, 1), 1); hxy = nd.sobel(nd.sobel(g, 0), 1)
        tr, det = hxx + hyy, xp.sqrt((hxx - hyy) ** 2 + 4 * hxy ** 2)
        F += [(tr + det) / 2, (tr - det) / 2]
    F.append(nd.median_filter(s[3], size=7))
    if xp is not np:
        F = [cp.asnumpy(f) for f in F]
    d = np.minimum(ndi.distance_transform_edt(~crys), DCAP).astype(np.float32)
    F.append(d)
    return np.stack(F, -1).astype(np.float32)


def crystal_from_image(g2d):
    c = ndi.gaussian_filter(g2d, 3) < 148
    c = ndi.binary_opening(c, iterations=3)
    lab, n = ndi.label(c)
    if n == 0:
        return c
    sz = ndi.sum(c, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.where(sz > 2000)[0])
    return ndi.binary_fill_holes(keep)


def labels_negative(name, meta):
    """crop with direct crystal-cell contact: tissue next to the crystal is cytoplasm, not vacuole"""
    m = meta[name]
    h, w = np.array(Image.open(A / name)).shape
    st = norm_stack(m["run"], m["z10"], m["y0"], m["y0"] + h, m["x0"], m["x0"] + w)
    g = ndi.gaussian_filter(st[3], 1)
    crys = crystal_from_image(st[3])
    tis = ndi.gaussian_filter(g, 3) < 193
    dc = ndi.distance_transform_edt(~crys)
    Y = np.full((h, w), 255, np.uint8)
    Y[ndi.binary_erosion(crys, iterations=3)] = 1
    Y[(dc > 3) & (dc < 60) & tis & ~crys] = 3
    Y[(ndi.gaussian_filter(g, 3) > 196) & (dc > 40)] = 0
    return st, crys, Y, np.zeros((h, w), bool)


def labels_from_lines(name, meta, ann):
    m = meta[name]
    img = np.array(Image.open(A / name))
    h, w = img.shape
    st = norm_stack(m["run"], m["z10"], m["y0"], m["y0"] + h, m["x0"], m["x0"] + w)
    g = ndi.gaussian_filter(st[3], 1)
    crys = crystal_from_image(st[3])
    Y = np.full((h, w), 255, np.uint8)
    a = ann[name]
    tis = ndi.gaussian_filter(g, 3) < 193
    def line_mask(k):
        mk = np.zeros((h, w), bool)
        if k in a:
            p = np.array(a[k]); mk[p[:, 0].clip(0, h - 1), p[:, 1].clip(0, w - 1)] = True
        return mk
    yel, blu, grn = line_mask("vacuole"), line_mask("cell"), line_mask("nucleus")
    Y[ndi.binary_erosion(crys, iterations=3)] = 1
    nucl = np.zeros((h, w), bool)
    if grn.any():
        nucl = ndi.binary_fill_holes(ndi.binary_dilation(grn, iterations=2)) & ~ndi.binary_dilation(grn, iterations=4)
        Y[nucl] = 4
    if yel.any():
        dc = ndi.distance_transform_edt(~crys)
        dv, ind = ndi.distance_transform_edt(~yel, return_indices=True)
        t = dc[ind[0], ind[1]]
        Y[(dc > 2) & (dc < t - 2) & (dv < 60) & ~crys] = 2  # any brightness: lucent lumen is vacuole too
        Y[(dc > t + 4) & (dc < t + 30) & (dv < 45) & tis & ~crys & ~ndi.binary_dilation(nucl | grn, iterations=4)
          & (Y == 255)] = 3
    if blu.any():
        db = ndi.distance_transform_edt(~blu)
        Y[(ndi.gaussian_filter(g, 3) > 195) & (db > 5) & ~crys] = 0
        # cytoplasm just inside the blue line (dark side), away from nuclei / crystal
        Y[(db > 4) & (db < 20) & tis & ~crys & (Y == 255) & ~ndi.binary_dilation(nucl, iterations=4)
          & (ndi.distance_transform_edt(~crys) > 40)] = 3
    return st, crys, Y, yel


def inside_cells(tis, crys):
    """tissue plus everything it encloses together with the crystal (lucent lumen between them)"""
    t = ndi.binary_closing(tis, iterations=2)
    return ndi.binary_fill_holes(t | crys) & ~crys


def predict_vacuole(prob, crys, inside):
    """vacuole = pixels classified vacuole or lucent (resin-like) inside cells, connected to the crystal surface"""
    am = prob.argmax(-1)
    v = ((am == 2) | (am == 0)) & ~crys & inside
    v = ndi.binary_opening(v, iterations=1)
    lab, n = ndi.label(v | crys)
    keep = np.unique(lab[crys]); keep = keep[keep > 0]
    return np.isin(lab, keep) & ~crys


def sample(F, Y, nper=15000, rng=np.random.default_rng(0)):
    X, y = [], []
    for k in range(len(CLS)):
        idx = np.flatnonzero(Y.ravel() == k)
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, min(nper, len(idx)), replace=False)
        X.append(F.reshape(-1, F.shape[-1])[idx]); y.append(np.full(len(idx), k))
    return np.concatenate(X), np.concatenate(y)


if __name__ == "__main__":
    mode = sys.argv[1]
    meta = json.load(open(A / "crops.json"))
    ann = json.load(open(A / "user_lines.json"))
    names = [n for n in ann if ann[n]] + [n for n in NEGATIVE if n in meta]
    data = {}
    for n in names:
        st, crys, Y, yel = labels_negative(n, meta) if n in NEGATIVE else labels_from_lines(n, meta, ann)
        data[n] = (features(st, crys), Y, crys, yel, st)
        log(n, "labels", {CLS[k]: int((Y == k).sum()) for k in range(len(CLS))})

    if mode == "eval":
        vac_names = [n for n in names if n in NEGATIVE or "vacuole" in ann.get(n, {})]
        res = {}
        for test in vac_names:
            Xs, ys = zip(*[sample(data[n][0], data[n][1]) for n in names if n != test])
            rf = RandomForestClassifier(150, min_samples_leaf=3, n_jobs=32, random_state=0).fit(np.concatenate(Xs), np.concatenate(ys))
            F, Y, crys, yel, st = data[test]
            prob = rf.predict_proba(F.reshape(-1, F.shape[-1])).reshape(*F.shape[:2], -1)
            full = np.zeros(prob.shape[:2] + (len(CLS),), np.float32); full[..., rf.classes_] = prob
            tis = ndi.gaussian_filter(st[3], 3) < 193
            v = predict_vacuole(full, crys, inside_cells(tis, crys))
            b = find_boundaries(v | crys, mode="outer") & ~crys
            d = ndi.distance_transform_edt(~b) if b.any() else np.full(v.shape, 999.)
            if test in NEGATIVE:
                surfn = crys & ~ndi.binary_erosion(crys)
                res[test] = dict(false_vacuole_um2_per_um_surface=float(v.sum() * 1e-4 / max(surfn.sum() * 1e-2, 1e-6)),
                                 surface_with_vacuole=float((ndi.binary_dilation(v, iterations=3) & surfn).sum() / max(surfn.sum(), 1)))
            else:
                p = np.argwhere(yel)
                dd = d[p[:, 0], p[:, 1]] * 10
                res[test] = dict(median_nm=float(np.median(dd)), p90_nm=float(np.percentile(dd, 90)), within30=float(np.mean(dd <= 30)))
            log(test, {k: round(v_, 2) for k, v_ in res[test].items()})
            o = np.stack([np.clip((st[3] - 110) / 105 * 255, 0, 255)] * 3, -1).astype(np.uint8)
            o[v] = (0.5 * o[v] + 0.5 * np.array((240, 228, 66))).astype(np.uint8)
            o[yel] = (0, 0, 0)
            Image.fromarray(o).save(ROOT / f"03_Output/run0910/explore/rf_loco_{test}")
        json.dump(res, open(ROOT / "03_Output/run0910/explore/rf_loco.json", "w"), indent=1)

    if mode == "apply":
        import pickle
        run = sys.argv[2]
        D = proc_dir(run)
        Xs, ys = zip(*[sample(data[n][0], data[n][1]) for n in names])
        rf = RandomForestClassifier(150, min_samples_leaf=3, n_jobs=32, random_state=0).fit(np.concatenate(Xs), np.concatenate(ys))
        pickle.dump(rf, open(D / "rf_vacuole.pkl", "wb"))
        c = json.load(open(D / "crop10.json")); lo, n10 = c["lo10"], c["shape10"]
        L = load(D / "lab10")
        crA = L == 2
        vac2d = np.zeros(L.shape, bool)
        for z in range(n10[0]):
            cr = crA[z]
            if not cr.any():
                continue
            dcr = ndi.distance_transform_edt(~cr)
            yy, xx = np.where(dcr <= 62)
            y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
            crs = cr[y0:y1, x0:x1]
            inside = ndi.binary_fill_holes((L[z, y0:y1, x0:x1] >= 10) | crs) & ~crs
            zone = inside & (dcr[y0:y1, x0:x1] <= 60)
            if not zone.any():
                continue
            st = norm_stack(run, z + lo[0], y0 + lo[1], y1 + lo[1], x0 + lo[2], x1 + lo[2])
            F = features(st, crs)
            pr = np.zeros(zone.shape + (len(CLS),), np.float32)
            pr[..., 3] = 1  # outside the zone: treat as cytoplasm
            tmp = np.zeros((zone.sum(), len(CLS)), np.float32)
            tmp[:, rf.classes_] = rf.predict_proba(F[zone])
            pr[zone] = tmp
            vac2d[z, y0:y1, x0:x1] = predict_vacuole(pr, crs, inside)
            if z % 50 == 0:
                log("slice", z, int(vac2d[z].sum()))
        # 3D cleanup: Z-smoothing, keep parts connected to the crystal, add the 20 nm membrane layer inside cells
        import cupy as cp
        from cupyx.scipy import ndimage as cnd
        sm = np.zeros(L.shape, bool)
        for z0 in range(0, n10[0], 128):
            a0, a1 = max(z0 - 8, 0), min(z0 + 136, n10[0])
            s_ = cp.asnumpy(cnd.gaussian_filter(cp.asarray(vac2d[a0:a1], dtype=cp.float32), (2, 1, 1))) > 0.5
            sm[z0:z0 + 128] = s_[z0 - a0:z0 - a0 + 128]
        sm &= ~crA
        lab = cc3d.connected_components(sm | crA, connectivity=6)
        keep = np.unique(lab[crA]); keep = keep[keep > 0]
        lum = np.isin(lab, keep) & ~crA
        cell = L >= 10
        body = lum | crA
        vac = lum | ((edt.edt(~body, parallel=NT) <= 2) & cell)
        if (D / "vac10.npz").exists() and not (D / "vac10_rule.npz").exists():
            save(D / "vac10_rule", load(D / "vac10"))
        save(D / "vac10", vac)
        # surface stats: thickness = distance from the crystal of the vacuole's outer boundary
        dcr3 = edt.edt(~crA, parallel=NT)
        outer = vac & ~ndi.binary_erosion(vac | crA, iterations=1)
        th = dcr3[outer] * 10
        surf = crA & (edt.edt(crA, parallel=NT) <= 1)
        near_cell = edt.edt(~cell, parallel=NT) <= 6
        has_vac = edt.edt(~lum, parallel=NT) <= 3
        cov = surf & near_cell
        S = json.load(open(D / "crystal10_surface_stats.json"))
        S.update(vacuole_method="classifier trained on hand-traced lines (11)",
                 vacuole_lumen_volume_um3=float(lum.sum() * 1e-6),
                 vacuole_surface_fraction=float((cov & has_vac).sum() / max(surf.sum(), 1)),
                 vacuole_thickness_nm_median=float(np.median(th[th >= 30])) if (th >= 30).any() else 0.0,
                 vacuole_thickness_nm_p90=float(np.percentile(th[th >= 30], 90)) if (th >= 30).any() else 0.0)
        json.dump(S, open(D / "crystal10_surface_stats.json", "w"), indent=1)
        log({k: (round(v, 3) if isinstance(v, float) else v) for k, v in S.items()})
