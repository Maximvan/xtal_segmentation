#!/bin/bash
# rebuild meshes with 3D-regularised native edges (10 regularise) + refresh viewer JSON + shrink to < 16 MB
cd /srv/scratch/maximva/xtal_segmentation/scripts
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3
PY="../env/xtal/bin/python -W ignore"; N=../logs/NIGHT2
for r in largeXtal_cM run0910 run0110; do
  echo "$(date '+%d %H:%M') remesh-reg: $r running" >> $N
  $PY 10_mesh_sdf_surfacemap.py $r > ../logs/n7_${r}_10.log 2>&1 && $PY - $r <<'PY' && $PY shrink_glb.py $r 14 >> ../logs/n7_${r}_10.log 2>&1 && echo "$(date '+%d %H:%M') remesh-reg: $r DONE" >> $N || echo "$(date '+%d %H:%M') remesh-reg: $r FAILED" >> $N
import json, sys
r=sys.argv[1]; p=f'../03_Output/viewer/data_{r}.json'; d=json.load(open(p))
S=json.load(open(f'../02_Processed/{r}/crystal10_surface_stats.json')); E=json.load(open(f'../02_Processed/{r}/native_edges.json'))
a=json.load(open(f'../03_Output/{r}/explore/mesh_hq_areas_um2.json'))
S['crystal_surface_um2']=a.get('crystal', S.get('crystal_surface_um2')); S['edges']='crystal + sac edges refined at native resolution, 3D-regularised (22 + 10)'
d['surface']=S; d['volumes']['crystal']=round(E['crystal_um3_native'],2); d['volumes']['vacuole_crystal']=round(E['sac_um3_native'],2)
json.dump(d,open(p,'w'),indent=1)
PY
done
echo "$(date '+%d %H:%M') remesh-reg: all done" >> $N
