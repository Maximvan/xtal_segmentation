#!/bin/bash
# Overnight 2026-11-25: all 5 samples in 3D at 10 nm with the crystal/sac U-Net (13, Jasmijn's labels).
# Two RTX 3090 (PCI order!): GPU 5 = run0110 ; GPU 4 = new samples, then run0910. Progress: logs/NIGHT2
cd /srv/scratch/maximva/xtal_segmentation/scripts
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY="../env/xtal/bin/python -W ignore"
N=../logs/NIGHT2
st() { echo "$(date '+%d %H:%M') $*" >> $N; }
pipeline() {  # pipeline <gpu> <run> <steps...>
  local gpu=$1 run=$2; shift 2
  for S in "$@"; do
    local den=""; [ "$S" = "07_refine10_den" ] && { S=07_refine10; den=1; }
    st "$run $S running (GPU $gpu)"
    if ! CUDA_VISIBLE_DEVICES=$gpu DEN=$den $PY $S.py $run > ../logs/n2_${run}_$S.log 2>&1; then
      st "$run $S FAILED (logs/n2_${run}_$S.log) - skipping rest of $run"; return 1; fi
  done
  st "$run DONE"
}
FULL="14_net_crystal_sac 02_segment_classical 07_refine10 16_merge_net 12_vacuole_constrain 05_mito_and_meshes 06_qc_overlays 10_mesh_sdf_surfacemap"
echo "night2 started $(date)" > $N
# wait for the U-Net (13: training + tests)
while pgrep -f "[1]3_sac_unet.py" > /dev/null; do sleep 30; done
[ -e ../02_Processed/sac_unet/sac_unet.pt ] || { st "no trained U-Net - abort"; exit 1; }
st "U-Net ready; tests in 03_Output/sac_unet/eval.json"
(
  pipeline 5 run0110 03_mitonet $FULL
) &
(
  for R in smallXtal_6D2P largeXtal_cM smallXtal_6M; do
    while pgrep -f "[1]5_import_dataset.py $R " > /dev/null; do sleep 30; done
    [ -e ../01_Input/derived/${R}_10nm.tif ] || { st "$R import missing"; continue; }
    pipeline 4 $R 01_preprocess 03_mitonet $FULL
  done
  pipeline 4 run0910 14_net_crystal_sac 02_segment_classical 07_refine10_den 16_merge_net 12_vacuole_constrain 05_mito_and_meshes 06_qc_overlays 10_mesh_sdf_surfacemap
) &
wait
st "night2 finished"; du -sh ../02_Processed ../03_Output ../01_Input/derived >> $N
