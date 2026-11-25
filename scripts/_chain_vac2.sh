#!/bin/bash
cd /srv/scratch/maximva/xtal_segmentation/scripts
export CUDA_VISIBLE_DEVICES=5
PY=../env/xtal/bin/python
for S in 12_vacuole_constrain 05_mito_and_meshes 06_qc_overlays 10_mesh_sdf_surfacemap; do
  echo "$(date +%H:%M) vac-constrain $S running" >> ../logs/STATUS
  $PY -W ignore $S.py run0910 > ../logs/${S}_vac2.log 2>&1 || { echo "$(date +%H:%M) vac-constrain $S FAILED" >> ../logs/STATUS; exit 1; }
done
echo "$(date +%H:%M) vac-constrain done" >> ../logs/STATUS
