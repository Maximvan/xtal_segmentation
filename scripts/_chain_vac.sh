#!/bin/bash
cd /srv/scratch/maximva/xtal_segmentation/scripts
export CUDA_VISIBLE_DEVICES=5
while kill -0 61158 2>/dev/null; do sleep 10; done
PY=../env/xtal/bin/python
for S in 05_mito_and_meshes 06_qc_overlays 10_mesh_sdf_surfacemap; do
  echo "$(date +%H:%M) vac-rf $S running" >> ../logs/STATUS
  $PY -W ignore $S.py run0910 > ../logs/${S}_vacrf.log 2>&1 || { echo "$(date +%H:%M) vac-rf $S FAILED" >> ../logs/STATUS; exit 1; }
done
echo "$(date +%H:%M) vac-rf done" >> ../logs/STATUS
