#!/bin/bash
cd /srv/scratch/maximva/xtal_segmentation/scripts
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4
PY="../env/xtal/bin/python -W ignore"; N=../logs/NIGHT2; r=smallXtal_6M
until grep -q "$r DONE (v3b)" $N || grep -q "$r [0-9a-z_]* FAILED (logs/n5" $N; do sleep 60; done
grep -q "$r DONE (v3b)" $N || exit 0
echo "$(date '+%d %H:%M') native-edges: $r 22 running" >> $N
$PY 22_native_edges.py $r > ../logs/n6_${r}_22.log 2>&1 && $PY 10_mesh_sdf_surfacemap.py $r > ../logs/n6_${r}_10.log 2>&1 \
  && echo "$(date '+%d %H:%M') native-edges: $r DONE" >> $N || echo "$(date '+%d %H:%M') native-edges: $r FAILED" >> $N
