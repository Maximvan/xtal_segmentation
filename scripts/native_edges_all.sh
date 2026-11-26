#!/bin/bash
# native edge refinement (22) + meshes (10) per sample, after its main run is done
cd /srv/scratch/maximva/xtal_segmentation/scripts
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4
PY="../env/xtal/bin/python -W ignore"; N=../logs/NIGHT2
st() { echo "$(date '+%d %H:%M') native-edges: $*" >> $N; }
do_run() { local r=$1
  st "$r 22 running"
  $PY 22_native_edges.py $r > ../logs/n6_${r}_22.log 2>&1 && $PY 10_mesh_sdf_surfacemap.py $r > ../logs/n6_${r}_10.log 2>&1 \
    && $PY - $r <<'PY' && st "$r DONE" || st "$r FAILED"
import json, sys
r=sys.argv[1]; p=f'../03_Output/viewer/data_{r}.json'; d=json.load(open(p))
S=json.load(open(f'../02_Processed/{r}/crystal10_surface_stats.json')); E=json.load(open(f'../02_Processed/{r}/native_edges.json'))
a=json.load(open(f'../03_Output/{r}/explore/mesh_hq_areas_um2.json'))
S['crystal_surface_um2']=a.get('crystal', S.get('crystal_surface_um2')); S['edges']='crystal + sac edges refined at native resolution (22)'
d['surface']=S; d['volumes']['crystal']=round(E['crystal_um3_native'],2); d['volumes']['vacuole_crystal']=round(E['sac_um3_native'],2)
json.dump(d,open(p,'w'),indent=1)
PY
}
do_run run0910
do_run largeXtal_cM
for r in run0110 smallXtal_6M; do
  until grep -q "$r DONE (v3" $N || grep -q "$r [0-9a-z_]* FAILED (logs/n[45]" $N; do sleep 60; done
  grep -q "$r DONE (v3" $N && do_run $r
done
st "all done"
