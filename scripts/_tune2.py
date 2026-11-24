import numpy as np, tifffile
from scipy import ndimage as ndi
from PIL import Image
from skimage.segmentation import find_boundaries
from skimage.morphology import disk
S='/tmp/claude-3287/-srv-scratch-maximva/aee6c328-5d10-4aa9-8149-a5261eecf715/scratchpad'
D='02_Processed/run0910/'
v=tifffile.imread(D+'img20.tif'); cr=np.load(D+'crystal.npy'); ok=np.load(D+'ok.npy')
rows=[]
for z in [100,150,260]:
    s=v[z].astype(np.float32)
    g=ndi.gaussian_filter(s,1.5)
    t=(g<191)&ok[z]
    t=ndi.binary_closing(t,disk(2)); t=ndi.binary_fill_holes(t); t=ndi.binary_opening(t,disk(2))
    g6=ndi.gaussian_filter(s,6)
    core=(g6>146)&(g6<168)&t&~ndi.binary_dilation(cr[z],disk(4))
    core=ndi.binary_opening(core,disk(9))
    # grow back to the envelope at finer scale
    g3=ndi.gaussian_filter(s,3)
    n=ndi.binary_dilation(core,disk(6))&(g3<176)&t&~cr[z]
    n=ndi.binary_closing(n,disk(3)); n=ndi.binary_fill_holes(n); n=ndi.binary_opening(n,disk(4))
    rgb=np.stack([v[z]]*3,-1).copy()
    rgb[find_boundaries(t)]=(255,128,0); rgb[find_boundaries(n)]=(255,0,255); rgb[find_boundaries(cr[z])]=(0,0,255)
    rows.append(rgb)
Image.fromarray(np.vstack(rows)).save(S+'/tune.png')
