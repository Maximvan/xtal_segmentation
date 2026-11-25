#!/bin/bash
# after the nucleus fix in 02: 10 nm redraw on denoised data -> vacuole classifier -> constraint -> meshes/QC/map
cd /srv/scratch/maximva/xtal_segmentation/scripts
export CUDA_VISIBLE_DEVICES=5
PY="../env/xtal/bin/python -W ignore"
st() { echo "$(date +%H:%M) nucfix $*" >> ../logs/STATUS; }
rm -f ../02_Processed/run0910/vac10_unconstrained.npz  # 12 must take the new 11 output
for S in "07_refine10.py run0910" "11_vacuole_classifier.py apply run0910" "12_vacuole_constrain.py run0910" \
         "05_mito_and_meshes.py run0910" "06_qc_overlays.py run0910" "10_mesh_sdf_surfacemap.py run0910"; do
  n=${S%%.py*}; st "$n running"
  if [ "$n" = "07_refine10" ]; then DEN=1 $PY $S > ../logs/${n}_nucfix.log 2>&1; else $PY $S > ../logs/${n}_nucfix.log 2>&1; fi \
    || { st "$n FAILED"; exit 1; }
done
st "done"
