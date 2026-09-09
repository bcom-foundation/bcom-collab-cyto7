import time; t0=time.time()
import numpy as np, nibabel as nib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm
from matplotlib.colors import Normalize
ROOT="."
SURF=ROOT+"/resources/fsaverage_surfaces/{h}.inflated"
CONF=ROOT+"/resources/cyto7_derived/cache_conf_v9/pial.{h}.cyto7.confidence{sfx}.shape.gii"
ANN=ROOT+"/resources/cyto7_derived/pial.{h}.cyto7.v9.annot"
MYE=ROOT+"/resources/neuromaps_cache/myelin_fsaverage_164k_{h}.npy"
OUT=ROOT+"/figures/v9/support/support_map_myelin.png"
LIGHT_BLUE=(0.68,0.85,0.90)
panels=[("lh","lateral"),("lh","medial"),("rh","lateral"),("rh","medial")]
def gii(p): return np.asarray(nib.load(p).darrays[0].data,float)
def proj(h,view,coords):
    y,z=coords[:,1],coords[:,2]
    return (-y,z) if (h,view) in (("lh","lateral"),("rh","medial")) else (y,z)
# precompute geometry per (h,view)
G={}
labs={h:np.asarray(nib.freesurfer.io.read_annot(ANN.format(h=h))[0]) for h in ("lh","rh")}
for h,view in panels:
    coords,faces=nib.freesurfer.read_geometry(SURF.format(h=h))
    tri=coords[faces]; n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); n/=np.linalg.norm(n,axis=1,keepdims=True)+1e-9
    fr=(n[:,0]<0) if (h,view) in (("lh","lateral"),("rh","medial")) else (n[:,0]>0)
    F=faces[fr]; fx=coords[F].mean(1)[:,0]
    order=np.argsort(-fx) if (h,view) in (("lh","lateral"),("rh","medial")) else np.argsort(fx)
    F=F[order]; sx,sy=proj(h,view,coords); verts=np.stack([sx[F],sy[F]],axis=-1)
    sh=0.62+0.38*np.clip(np.abs(n[fr,2][order]),0,1)
    med=(labs[h][F]==0).sum(1)>=2
    G[(h,view)]=dict(F=F,verts=verts,sh=sh,med=med,lo=(sx.min(),sy.min()),hi=(sx.max(),sy.max()))
# rows
comb={h:gii(CONF.format(h=h,sfx="")) for h in ("lh","rh")}
comps={c:{h:gii(CONF.format(h=h,sfx="_"+c)) for h in ("lh","rh")} for c in ("atlas","topo","geom","prior")}
mye={h:np.load(MYE.format(h=h)) for h in ("lh","rh")}
# robust myelin range over cortex
allm=np.concatenate([mye[h][(labs[h]>0)&(mye[h]>0)] for h in ("lh","rh")])
mvmin,mvmax=np.percentile(allm,2),np.percentile(allm,98)
rows=[("combined (anatomy)",comb,"viridis",0,1,"conf"),
      ("atlas",comps["atlas"],"viridis",0,1,"conf"),
      ("topo",comps["topo"],"viridis",0,1,"conf"),
      ("geom",comps["geom"],"viridis",0,1,"conf"),
      ("prior",comps["prior"],"viridis",0,1,"conf"),
      ("T1w/T2w myelin\n(reference, NOT in score)",mye,"magma",mvmin,mvmax,"mye")]
nrow,ncol=len(rows),4
fig,axes=plt.subplots(nrow,ncol,figsize=(ncol*3.2,nrow*2.5)); fig.patch.set_facecolor("white")
for r,(name,data,cmn,vmin,vmax,kind) in enumerate(rows):
    cmap=plt.get_cmap(cmn); nm=Normalize(vmin,vmax)
    for c,(h,view) in enumerate(panels):
        ax=axes[r,c]; g=G[(h,view)]; vals=data[h]
        fv=np.nanmean(vals[g["F"]],axis=1)
        col=cmap(nm(fv))[:,:3]
        bad=g["med"]|~np.isfinite(fv)
        if kind=="mye": bad=bad|(fv<=0)
        col[bad]=LIGHT_BLUE
        col=np.clip(col*g["sh"][:,None],0,1)
        ax.add_collection(PolyCollection(g["verts"],facecolors=col,edgecolors="face",linewidths=0.4))
        ax.set_xlim(g["lo"][0]-2,g["hi"][0]+2); ax.set_ylim(g["lo"][1]-2,g["hi"][1]+2)
        ax.set_aspect("equal"); ax.axis("off")
        if r==0: ax.set_title(f"{h.upper()} {view}",fontsize=12)
        if c==0: ax.text(-0.02,0.5,name,transform=ax.transAxes,rotation=90,va="center",ha="center",fontsize=11,weight="bold")
# two colorbars
sm1=cm.ScalarMappable(cmap="viridis",norm=Normalize(0,1))
cb1=fig.colorbar(sm1,ax=axes[:5,:].ravel().tolist(),fraction=0.02,pad=0.01); cb1.set_label("support (anatomy)",fontsize=11)
sm2=cm.ScalarMappable(cmap="magma",norm=Normalize(mvmin,mvmax))
cb2=fig.colorbar(sm2,ax=axes[5,:].ravel().tolist(),fraction=0.02,pad=0.01); cb2.set_label("T1w/T2w",fontsize=11)
fig.suptitle("cyto7 per-vertex support: combined + components (T1w/T2w shown for reference)",fontsize=15,y=0.995)
fig.subplots_adjust(left=0.04,right=0.9,top=0.96,bottom=0.01,wspace=0.0,hspace=0.05)
fig.savefig(OUT,dpi=115,facecolor="white",bbox_inches="tight"); print("saved",OUT,"in",round(time.time()-t0,1),"s")
print(f"myelin range 2-98pct: {mvmin:.2f}-{mvmax:.2f}")
