# xtal segmentation

3D segmentation of FIB-SEM neutrophils around galectin-10 (Charcot-Leyden) crystals:
cells, nuclei, organelles, crystal vacuole and the crystal itself (runs 0110 and 0910).

- `scripts/` — pipeline (`01_preprocess` → `02_segment_classical` → `03_mitonet` → `07_refine10`
  → `05_mito_and_meshes` → `06_qc_overlays`; `run_all.sh` chains them). `_*.py` are scratch/tuning scripts.
- `03_Output/viewer/index.html` — 3D viewer page (data files are generated, not tracked)
- `requirements.txt` — pip freeze of the `env/xtal` venv

FIB-SEM data, processed volumes, model weights and outputs are **not** in git; they live on
`/srv/scratch/maximva/xtal_segmentation`.
