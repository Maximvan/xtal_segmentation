#!/bin/bash
cd /srv/scratch/maximva/xtal_segmentation/scripts
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
PY="../env/xtal/bin/python -W ignore"; N=../logs/NIGHT2
for s in 07_refine10 16_merge_net 12_vacuole_constrain 05_mito_and_meshes 06_qc_overlays 10_mesh_sdf_surfacemap; do
  echo "$(date '+%d %H:%M') smallXtal_6M $s running (GPU 2, v3b)" >> $N
  $PY $s.py smallXtal_6M > ../logs/n5_smallXtal_6M_$s.log 2>&1 || { echo "$(date '+%d %H:%M') smallXtal_6M $s FAILED (logs/n5_smallXtal_6M_$s.log)" >> $N; exit 1; }
done
echo "$(date '+%d %H:%M') smallXtal_6M DONE (v3b)" >> $N
