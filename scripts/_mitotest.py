import torch, numpy as np, tifffile, sys
from PIL import Image
S='/tmp/claude-3287/-srv-scratch-maximva/aee6c328-5d10-4aa9-8149-a5261eecf715/scratchpad'
m=torch.jit.load('env/models/MitoNet_v1.pth',map_location='cuda').eval()
def run(img):
    H,W=img.shape; ph=(16-H%16)%16; pw=(16-W%16)%16
    x=np.pad(img.astype(np.float32)/255.,((0,ph),(0,pw)),mode='reflect')
    x=(x-0.57571)/0.12765
    with torch.no_grad():
        o=m(torch.from_numpy(x)[None,None].cuda())
    print({k:tuple(v.shape) for k,v in o.items()})
    return torch.sigmoid(o['sem_logits'][0,0]).cpu().numpy()[:H,:W]
# 20 nm
v=tifffile.imread('02_Processed/run0910/img20.tif'); s=v[150]
p=run(s)
Image.fromarray(np.hstack([s,(p*255).astype(np.uint8)])[60:420, 900:1492*1]).save(S+'/mito20.png')
# 10 nm from bin2 raw, slice matching z=300 of raw (=150 in 20nm)
f='01_Input/FibSEM xtals/2026_10_01_JasmijnB_FIBSEM_CLEM1/run_0910/EM447_CLEM1_0910_bin2.tif'
with tifffile.TiffFile(f) as t: s2=t.pages[300].asarray()
print(s2.shape, s2.mean())
crop=s2[150:830, 1900:2990]
p2=run(crop)
Image.fromarray(np.hstack([crop,(p2*255).astype(np.uint8)])).save(S+'/mito10.png')
