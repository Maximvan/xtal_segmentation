#!/bin/bash
# 2026-11-26: re-import (fixed resampling) + destripe (per-slice contrast) + full pipeline, 4 RTX 3090 (PCI order)
cd /srv/scratch/maximva/xtal_segmentation/scripts
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY="../env/xtal/bin/python -W ignore"
S="../01_Input/FibSEM xtals"
N=../logs/NIGHT2
st() { echo "$(date '+%d %H:%M') $*" >> $N; }
step() { local gpu=$1 run=$2 s=$3; shift 3; st "$run $s running (GPU $gpu, v3)"
  CUDA_VISIBLE_DEVICES=$gpu $PY $s.py $run "$@" > ../logs/n4_${run}_$s.log 2>&1 || { st "$run $s FAILED (logs/n4_${run}_$s.log)"; return 1; }; }
FULL="01_preprocess 03_mitonet 14_net_crystal_sac 02_segment_classical 07_refine10 16_merge_net 12_vacuole_constrain 05_mito_and_meshes 06_qc_overlays 10_mesh_sdf_surfacemap"
pipe() { local gpu=$1 run=$2; for s in $FULL; do step $gpu $run $s || return 1; done; st "$run DONE (v3)"; }
imp() { local name=$1 src=$2 xy=$3 zn=$4
  st "$name import running (v3)"
  rm -f ../01_Input/derived/${name}_ds_*.tif
  $PY 15_import_dataset.py $name "$src" $xy $zn > ../logs/n4_${name}_15.log 2>&1 || { st "$name import FAILED"; return 1; }; }
echo "---- v3 $(date)" >> $N
( step 5 run0110 20_destripe && pipe 5 run0110 ) &
( step 4 run0910 20_destripe && pipe 4 run0910 ) &
( imp largeXtal_cM "$S/EM447_largeXtal_cM_1205.tif" 6 6 && step 3 largeXtal_cM 20_destripe && pipe 3 largeXtal_cM ) &
( imp smallXtal_6M "$S/EM447_CLEM2_smallXtal_6M_0408/EM447_CLEM2_SmallX_6m.tif" 8 8 && step 2 smallXtal_6M 01_preprocess \
  && step 2 smallXtal_6M 14_net_crystal_sac && step 2 smallXtal_6M 20_destripe && pipe 2 smallXtal_6M ) &
( imp smallXtal_6D2P "$S/EM447_CLEM2_SmallXtal_6D2P_0106/EM447_CLEM2_6D2P_010626.tif" 6 6 ) &
wait
st "v3 finished"
