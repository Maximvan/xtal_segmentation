import numpy as np, tifffile, sys, os
from PIL import Image
from skimage.segmentation import find_boundaries
S='/tmp/claude-3287/-srv-scratch-maximva/aee6c328-5d10-4aa9-8149-a5261eecf715/scratchpad'
import sys
D='02_Processed/'+(sys.argv[1] if sys.argv[1].startswith('run') else 'run0910')+'/'
if sys.argv[1].startswith('run'): sys.argv.pop(1)
v=tifffile.imread(D+'img20.tif'); cr=np.load(D+'crystal.npy'); nu=np.load(D+'nuclei.npy'); ce=np.load(D+'cells.npy')
vx=np.load(D+'vac_xtal.npy') if os.path.exists(D+'vac_xtal.npy') else np.zeros(cr.shape,bool); vo=np.load(D+'vac_other.npy')|np.load(D+'granules.npy')
import os
mp=np.load(D+'mito_prob20.npy') if os.path.exists(D+'mito_prob20.npy') else np.zeros((1,1,1),np.uint8); mp=np.pad(mp,((0,0),(0,v.shape[1]-mp.shape[1]),(0,0)))
cols={1:(230,159,0),2:(86,180,233),3:(0,158,115),4:(240,228,66),5:(204,121,167),6:(213,94,0)}
def ov(sl):
    im,c,n,cl,a,b,m=[x[sl] for x in (v,cr,nu,ce,vx,vo,mp)]
    rgb=np.stack([im]*3,-1).astype(np.float32)
    for k,col in cols.items():
        rgb[cl==k]=0.7*rgb[cl==k]+0.3*np.array(col)
    rgb[find_boundaries(c)]=(0,0,255)
    rgb[find_boundaries(n>0)]=(255,0,255)
    rgb[a]=0.5*rgb[a]+0.5*np.array((255,255,0))
    rgb[b]=(0,255,255)
    rgb[m>128]=(0,200,0)
    return rgb.astype(np.uint8)
zs=[int(z) for z in sys.argv[1:]] or [100,150,200,260]
Image.fromarray(np.vstack([ov(np.s_[z]) for z in zs])).save(S+'/ov.png')
Image.fromarray(ov(np.s_[:,229,:])).save(S+'/ov_xz.png')
