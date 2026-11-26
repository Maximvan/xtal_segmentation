#!/bin/bash
# 2026-11-26 reruns: smallXtal_6M from 14 (memory-light cleanup), run0910 from 07 (raw input; den10 crop is stale)
cd /srv/scratch/maximva/xtal_segmentation/scripts
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY="../env/xtal/bin/python -W ignore"
N=../logs/NIGHT2
st() { echo "$(date '+%d %H:%M') $*" >> $N; }
pipeline() { local gpu=$1 run=$2; shift 2
  for S in "$@"; do st "$run $S running (GPU $gpu, rerun)"
    CUDA_VISIBLE_DEVICES=$gpu $PY $S.py $run > ../logs/n3_${run}_$S.log 2>&1 || { st "$run $S FAILED (logs/n3_${run}_$S.log)"; return 1; }
  done; st "$run DONE (rerun)"; }
pipeline 4 smallXtal_6M 14_net_crystal_sac 02_segment_classical 07_refine10 16_merge_net 12_vacuole_constrain 05_mito_and_meshes 06_qc_overlays 10_mesh_sdf_surfacemap &
pipeline 5 run0910 07_refine10 16_merge_net 12_vacuole_constrain 05_mito_and_meshes 06_qc_overlays 10_mesh_sdf_surfacemap &
wait
