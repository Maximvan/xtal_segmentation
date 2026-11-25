#!/bin/bash
cd /srv/scratch/maximva/xtal_segmentation/scripts
export CUDA_VISIBLE_DEVICES=5
# (no wait)
PY=../env/xtal/bin/python
for STEP in 07_refine10 05_mito_and_meshes 06_qc_overlays; do
  echo "$(date +%H:%M) run0910 $STEP running" >> ../logs/STATUS
  $PY $STEP.py run0910 > ../logs/${STEP}_run0910.log 2>&1 || { echo "$(date +%H:%M) run0910 $STEP FAILED" >> ../logs/STATUS; exit 1; }
done
echo "$(date +%H:%M) run0910 done" >> ../logs/STATUS
