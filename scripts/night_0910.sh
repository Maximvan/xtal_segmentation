#!/bin/bash
# Overnight exploration for run0910 (run0110 untouched). Detached: survives logout.
# Start: setsid nohup bash scripts/night_0910.sh > logs/night.log 2>&1 < /dev/null &
# Progress: logs/NIGHT (one line per step), per-step logs in logs/night_*.log
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU:-5}
PY=../env/xtal/bin/python
L=../logs
P=../02_Processed/run0910
N=$L/NIGHT
step() { echo "$(date '+%d %H:%M') $*" >> $N; }
run() {  # run <name> <cmd...>
  local name=$1; shift
  step "$name running"
  if "$@" > $L/night_$name.log 2>&1; then step "$name ok"; else step "$name FAILED (logs/night_$name.log)"; fi
}
echo "night started $(date)" > $N

# 0. wait for the v4 run (tuned vacuole) to finish
while ! sed -n '/---- 10 nm v4/,$p' $L/STATUS | grep -q "run0910 done\|FAILED"; do sleep 60; done
step "v4 finished: $(sed -n '/---- 10 nm v4/,$p' $L/STATUS | tail -1)"

# keep the v4 (raw-input) result for comparison
mkdir -p $P/v4_raw ../03_Output/run0910/v4_raw
cp $P/lab10.npz $P/vac10.npz $P/mito10.npz $P/gran10.npz $P/crystal10_surface_stats.json $P/v4_raw/ 2>/dev/null
cp $P/dense10.npz $P/v4_raw/ 2>/dev/null
cp ../03_Output/viewer/scene_run0910.glb.txt ../03_Output/viewer/data_run0910.json ../03_Output/run0910/v4_raw/
cp -r ../03_Output/run0910/qc ../03_Output/run0910/v4_raw/
step "v4 outputs copied to 02_Processed/run0910/v4_raw and 03_Output/run0910/v4_raw"

# 1. self-supervised denoising (GPU)  |  3+4. SDF meshes + crystal surface map on v4 (CPU, parallel)
run 10_sdf_v4 $PY 10_mesh_sdf_surfacemap.py run0910 &
CPU=$!
run 08_denoise $PY 08_denoise_n2v.py run0910 20000
wait $CPU
cp -r ../03_Output/run0910/explore ../03_Output/run0910/v4_raw/explore_v4 2>/dev/null

# 2. redraw everything at 10 nm on the denoised data (+ nucleus >= 20 nm inside the cell)
if [ -e $P/den10.npz ]; then
  export DEN=1
  for S in 07_refine10 05_mito_and_meshes 06_qc_overlays; do run ${S}_den $PY $S.py run0910; done
  unset DEN
  mkdir -p ../03_Output/run0910/explore/qc_den && cp ../03_Output/run0910/qc/*.png ../03_Output/run0910/explore/qc_den/
  run 10_sdf_den $PY 10_mesh_sdf_surfacemap.py run0910
fi

# 3. U-Net on our own labels (pseudo-labels), trained on the denoised data (GPU)
run 09_unet $PY 09_pseudo_unet.py run0910 30000

step "night finished"
du -sh ../02_Processed ../03_Output >> $N
