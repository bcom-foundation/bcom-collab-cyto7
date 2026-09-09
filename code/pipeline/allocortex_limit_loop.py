"""Display overlay: allocortex as the hemisphere's topological limit (closed loop),
exposed as an open arc on the surface. Overlay only — released map unchanged.
Usage: python scripts/allocortex_limit_loop.py [v8]   (nibabel+matplotlib, no nilearn)
"""
import os, sys
import numpy as np, nibabel as nib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURF=ROOT+"/resources/fsaverage_surfaces/{h}.inflated"; ANN=ROOT+"/resources/cyto7_derived/pial.{h}.cyto7.{v}.annot"
V=sys.argv[1] if len(sys.argv)>1 else "v8"
OUT=ROOT+f"/figures/v9/anatomy3d/allocortex_limit_loop_{V}.png"
CY={0:(0.86,0.92,0.98),1:(0.55,0.20,0.65),2:(0.30,0.45,0.88),3:(0.30,0.70,0.75),4:(0.45,0.78,0.52),5:(0.75,0.85,0.45),6:(0.98,0.72,0.35),7:(0.98,0.88,0.45)}
NM={1:"Allocortex",2:"Agranular",3:"Dysgranular",4:"Eulaminate I",5:"Eulaminate II",6:"Eulaminate III",7:"Koniocortex"}
def proj(coords,h):
    P=coords[:,[1,2]].astype(float)
    if h=="lh": P=P.copy(); P[:,0]=-P[:,0]
    return P
def panel(ax,h):
    coords,faces=nib.freesurfer.read_geometry(SURF.format(h=h)); lab=np.asarray(nib.freesurfer.io.read_annot(ANN.format(h=h,v=V))[0])
    tri=coords[faces]; n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); n/=np.linalg.norm(n,axis=1,keepdims=True)+1e-9
    keep = n[:,0]>0.10 if h=="lh" else n[:,0]<-0.10
    F=faces[keep]; fl=lab[F]; fmaj=np.array([np.bincount(r[r>=0],minlength=8).argmax() if len(r) else 0 for r in fl])
    sh=0.65+0.35*np.abs(n[keep,0]); base=np.array([CY.get(int(t),(0.85,0.85,0.85)) for t in fmaj])
    rgb=np.clip(base*sh[:,None],0,1); rgb=0.55*rgb+0.45*np.ones_like(rgb); allo_f=fmaj==1; rgb[allo_f]=np.clip(base[allo_f]*sh[allo_f,None],0,1)
    P=proj(coords,h); ax.add_collection(PolyCollection(P[F],facecolors=rgb,edgecolors="none",linewidths=0))
    e=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[0,2]]]); e=np.unique(np.sort(e,1),axis=0)
    bnd=(lab[e[:,0]]==0)^(lab[e[:,1]]==0); eb=e[bnd]
    vset=np.zeros(lab.shape,bool); vset[F.ravel()]=True; eb=eb[vset[eb[:,0]]&vset[eb[:,1]]]
    cortlab=np.where(lab[eb[:,0]]>0,lab[eb[:,0]],lab[eb[:,1]]); seg=np.stack([P[eb[:,0]],P[eb[:,1]]],axis=1); allo=cortlab==1
    ax.add_collection(LineCollection(seg[~allo],colors=[(0.35,0.35,0.40,0.85)],linewidths=1.1,linestyles='dashed'))
    ax.add_collection(LineCollection(seg[allo],colors=[(0.50,0.0,0.65,1.0)],linewidths=3.2))
    ax.autoscale(); ax.set_aspect("equal"); ax.axis("off"); xl=ax.get_xlim(); zl=ax.get_ylim()
    ax.text(xl[0]+3,zl[0]+3,"anterior",fontsize=9,style="italic",color="0.3"); ax.text(xl[1]-3,zl[0]+3,"posterior",fontsize=9,style="italic",color="0.3",ha="right")
    ax.set_title(f"{h.upper()} — medial view ({V})",fontsize=12,weight="bold")
fig,axes=plt.subplots(1,2,figsize=(16,7)); fig.patch.set_facecolor("white")
for ax,h in zip(axes,("lh","rh")): panel(ax,h)
handles=[Line2D([0],[0],color=(0.50,0.0,0.65),lw=3.2,label="allocortex exposed ON the surface (β1=0 arc)"),
         Line2D([0],[0],color=(0.35,0.35,0.40),lw=1.0,ls='--',label="hemisphere limit — allocortical rim, closes OFF-surface (hippocampus, indusium griseum)")]
handles+=[Patch(facecolor=CY[c],label=NM[c]) for c in range(1,8)]+[Patch(facecolor=CY[0],label="medial wall")]
fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=8.5,frameon=False,bbox_to_anchor=(0.5,-0.05))
fig.suptitle(f"Allocortex as the hemisphere's topological limit ({V}) — a CLOSED loop (thin), exposed as an ARC on the surface (bold)",fontsize=13,weight="bold",y=1.0)
fig.text(0.5,-0.10,"Display overlay only — the released 7-label map is unchanged (allocortex = the bold arc). Thin closed contour = cortex/medial-wall boundary = topological limit; "
         "the allocortical ring completes along it through off-surface archicortex ('cupcake' growth).",ha="center",fontsize=8.5,style="italic",color="0.35",wrap=True)
fig.subplots_adjust(bottom=0.14,top=0.9,wspace=0.03)
fig.savefig(OUT,dpi=155,facecolor="white",bbox_inches="tight"); print("saved",OUT)
