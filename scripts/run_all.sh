#!/bin/bash
# Full pipeline, detached from the terminal. One RTX 3090. Runs given as args (default run0910;
# run0110 is on hold until Jasmijn's annotations arrive).
# Start:   setsid nohup bash scripts/run_all.sh run0910 > logs/run_all.log 2>&1 < /dev/null &
# Status:  cat logs/STATUS ; tail logs/run_all.log ; per-step logs in logs/<step>_<run>.log
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU:-5}
PY=../env/xtal/bin/python
L=../logs
echo "started $(date)" > $L/STATUS

for RUN in ${@:-run0910}; do
  # convert any legacy uncompressed arrays first (saves disk)
  $PY _compress.py ../02_Processed/$RUN > $L/compress_$RUN.log 2>&1
  [ -e ../02_Processed/$RUN/mito_prob20.npz ] || $PY 03_mitonet.py $RUN > $L/03_$RUN.log 2>&1
  ok=1
  for STEP in 02_segment_classical 07_refine10 05_mito_and_meshes 06_qc_overlays; do
    echo "$(date +%H:%M) $RUN $STEP running" >> $L/STATUS
    if ! $PY $STEP.py $RUN > $L/${STEP}_$RUN.log 2>&1; then
      echo "$(date +%H:%M) $RUN $STEP FAILED (see logs/${STEP}_$RUN.log)" >> $L/STATUS
      ok=0; break
    fi
  done
  [ $ok = 1 ] && echo "$(date +%H:%M) $RUN done" >> $L/STATUS
  $PY _compress.py ../02_Processed/$RUN >> $L/compress_$RUN.log 2>&1
done
echo "finished $(date)" >> $L/STATUS
du -sh ../02_Processed ../03_Output >> $L/STATUS
