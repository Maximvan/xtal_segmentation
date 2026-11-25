"""12 - Biological constraint on the crystal vacuole: it is a compartment of a cell.

A vacuole voxel is kept only if
  (a) the crystal-surface point it belongs to (nearest surface voxel) is covered by a cell: some cell voxel
      that belongs to the same surface point lies within COVER_NM of the crystal, and
  (b) it is not extracellular resin (the large resin region connected around the cells in 3D).
Then only parts touching the crystal are kept (the 20 nm membrane layer from 11 is kept as is).
usage: python 12_vacuole_constrain.py run0910       (input vac10 from 11; raw copy kept as vac10_unconstrained)
"""
import sys, json, time
import numpy as np, edt, cc3d
from scipy import ndimage as ndi
from xtal_lib import load, save, proc_dir, NT

run = sys.argv[1]
D = proc_dir(run)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
COVER = 70  # px (700 nm): a cell must lie within this distance of the crystal along the same surface point

L = load(D / "lab10")
src = D / "vac10_unconstrained.npz"
vac = load(src if src.exists() else D / "vac10").astype(bool)
if not src.exists():
    save(D / "vac10_unconstrained", vac)
cr = L == 2
cell = L >= 10
# (b) extracellular resin: largest 3D component of resin
rl = cc3d.connected_components(L == 1, connectivity=6)
cnt = cc3d.statistics(rl)["voxel_counts"]; cnt[0] = 0
extra = rl == cnt.argmax()
del rl
log("extracellular resin vox", int(extra.sum()), "vacuole vox in it", int((vac & extra).sum()))
vac &= ~extra

# (a) coverage per surface point, in Z chunks with halo (nearest-surface ownership needs <= COVER px context)
keep = np.zeros(L.shape, bool)
Z = L.shape[0]
CH, H = 96, COVER + 2
for z0 in range(0, Z, CH):
    a0, a1 = max(z0 - H, 0), min(z0 + CH + H, Z)
    c = cr[a0:a1]
    if not c.any():
        continue
    zz, yy, xx = np.where(edt.edt(~c, parallel=NT) <= COVER)
    bb = np.s_[zz.min():zz.max() + 1, yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    cb, cellb, vb = c[bb], cell[a0:a1][bb], vac[a0:a1][bb]
    dcr, ind = ndi.distance_transform_edt(~cb, return_indices=True)
    owner = np.ravel_multi_index(ind, cb.shape)  # nearest crystal voxel (a surface voxel) per voxel
    del ind
    near = (dcr > 0) & (dcr <= COVER)
    covered = np.zeros(cb.size, bool)
    covered[owner[cellb & near]] = True
    ok = vb & covered[owner]
    # write back only the chunk core (halo slices may have truncated ownership)
    zb0 = bb[0].start + a0  # absolute z of ok[0]
    c0, c1 = max(z0, zb0), min(z0 + CH, zb0 + ok.shape[0])
    if c1 > c0:
        keep[c0:c1, bb[1], bb[2]] |= ok[c0 - zb0:c1 - zb0]
    log(f"chunk {z0}", int(ok.sum()))
    del dcr, owner, near, covered, ok
vac &= keep

# keep parts touching the crystal; redraw the membrane layer inside cells
lum = vac & ~cr
lab = cc3d.connected_components(lum | cr, connectivity=6)
ids = np.unique(lab[cr]); ids = ids[ids > 0]
lum = np.isin(lab, ids) & ~cr
del lab
vac = lum  # input from 11 already includes the 20 nm membrane layer: do not add another
save(D / "vac10", vac)
# lumen for statistics = vacuole minus its 20 nm membrane shell inside cells
lum = vac & ~((edt.edt(vac | cr, parallel=NT) <= 2) & cell)

# stats
dcr3 = edt.edt(~cr, parallel=NT)
outer = vac & ~ndi.binary_erosion(vac | cr, iterations=1)
th = dcr3[outer] * 10
surf = cr & (edt.edt(cr, parallel=NT) <= 1)
cov = surf & (edt.edt(~cell, parallel=NT) <= 6)
has_vac = edt.edt(~lum, parallel=NT) <= 3
S = json.load(open(D / "crystal10_surface_stats.json"))
S.update(vacuole_method="classifier trained on hand-traced lines (11) + cell-coverage constraint (12)",
         vacuole_lumen_volume_um3=float(lum.sum() * 1e-6),
         vacuole_surface_fraction=float((surf & has_vac).sum() / max(surf.sum(), 1)),
         vacuole_thickness_nm_median=float(np.median(th[th >= 30])) if (th >= 30).any() else 0.0,
         vacuole_thickness_nm_p90=float(np.percentile(th[th >= 30], 90)) if (th >= 30).any() else 0.0,
         vacuole_on_bare_crystal_fraction=float((surf & has_vac & ~cov).sum() / max((surf & has_vac).sum(), 1)))
json.dump(S, open(D / "crystal10_surface_stats.json", "w"), indent=1)
log({k: (round(v, 3) if isinstance(v, float) else v) for k, v in S.items()})
