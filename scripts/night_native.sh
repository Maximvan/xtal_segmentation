#!/bin/bash
# Native-resolution crystal/sac (17 train+tests -> 18 apply per run -> 10 re-mesh), GPU 3 (PCI order).
cd /srv/scratch/maximva/xtal_segmentation/scripts
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3
PY="../env/xtal/bin/python -W ignore"
N=../logs/NIGHT2
st() { echo "$(date '+%d %H:%M') native: $*" >> $N; }
while pgrep -f "[1]7_native_sac.py" > /dev/null || pgrep -f "[1]3_sac_unet.py" > /dev/null; do sleep 30; done
[ -e ../03_Output/sac_unet_native/eval.json ] && [ -e ../03_Output/sac_unet/eval.json ] || { st "missing eval.json - native skipped"; exit 1; }
VERDICT=$($PY - <<'PY'
import json
a=json.load(open('../03_Output/sac_unet/eval.json'))['students_run0910']; b=json.load(open('../03_Output/sac_unet_native/eval.json'))['students_run0910']
ok = b['sac']['iou'] >= a['sac']['iou'] - 0.02 and b['crystal']['iou'] >= a['crystal']['iou'] - 0.02
print(('USE' if ok else 'SKIP') + f" native sac IoU {b['sac']['iou']:.3f} vs 10nm {a['sac']['iou']:.3f}; crystal {b['crystal']['iou']:.3f} vs {a['crystal']['iou']:.3f}")
PY
)
st "$VERDICT"
case "$VERDICT" in USE*) ;; *) exit 0;; esac
todo="smallXtal_6D2P largeXtal_cM run0110 smallXtal_6M run0910"
while [ -n "$todo" ]; do
  next=""
  for R in $todo; do
    if grep -q " $R DONE" $N; then
      st "$R 18_native_apply running"
      if $PY 18_native_apply.py $R > ../logs/n2_${R}_18.log 2>&1 && $PY 10_mesh_sdf_surfacemap.py $R > ../logs/n2_${R}_10native.log 2>&1; then
        $PY - $R <<'PY'
import json, sys
r=sys.argv[1]; p=f'../03_Output/viewer/data_{r}.json'; d=json.load(open(p))
S=json.load(open(f'../02_Processed/{r}/crystal10_surface_stats.json'))
a=json.load(open(f'../03_Output/{r}/explore/mesh_hq_areas_um2.json')); S['crystal_surface_um2']=a.get('crystal', S.get('crystal_surface_um2'))
d['surface']=S
if 'native_crystal_volume_um3' in S: d['volumes']['crystal']=round(S['native_crystal_volume_um3'],3); d['volumes']['vacuole_crystal']=round(S['native_sac_volume_um3'],3)
json.dump(d,open(p,'w'),indent=1)
PY
        st "$R native DONE"
      else st "$R native FAILED (logs/n2_${R}_18.log / _10native.log)"; fi
    elif grep -q " $R .*FAILED" $N; then st "$R skipped (10 nm pipeline failed)"
    else next="$next $R"; fi
  done
  todo=$(echo $next); [ -n "$todo" ] && sleep 120
  grep -q "night2 finished" $N && [ -n "$todo" ] && ! pgrep -f "[n]ight_all.sh" >/dev/null && { st "night2 ended; not done: $todo"; break; }
done
st "all native done"
