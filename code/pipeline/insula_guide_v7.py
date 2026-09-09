"""Insula painting guide for a given cyto7 version (default v7), inflated + lateral.

Fill = current cyto7 type; overlay (magenta mesh) = the von Economo frontoinsular
agranular target (FJK/FI) = where GC/von Economo place the agranular insula. Lets you
see the painted agranular sector (blue) against that target. nibabel + matplotlib only.

Usage:  python scripts/insula_guide_v7.py [v7]
"""
import os, sys
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURF=ROOT+"/resources/fsaverage_surfaces/{h}.inflated"
ANN=ROOT+"/resources/cyto7_derived/pial.{h}.cyto7.{v}.annot"
DEST=ROOT+"/resources/refine_atlases/{h}.aparc.a2009s.annot"
ECON=ROOT+"/resources/voneconomo/{h}.economo.annot"
VER=sys.argv[1] if len(sys.argv)>1 else "v7"
OUT=ROOT+f"/figures/v9/adjudication/insula_paint_guide_{VER}.png"
CYTO7={1:(0.55,0.20,0.65),2:(0.20,0.35,0.85),3:(0.10,0.62,0.70),4:(0.25,0.70,0.35),
       5:(0.62,0.78,0.20),6:(0.98,0.62,0.10),7:(0.98,0.85,0.15)}
NAMES={1:"Allocortex",2:"Agranular",3:"Dysgranular",4:"Eulaminate I",5:"Eulaminate II",6:"Eulaminate III",7:"Koniocortex"}
INS=["G_Ins_lg_and_S_cent_ins","G_insular_short","S_circular_insula_ant","S_circular_insula_inf","S_circular_insula_sup"]

def sub(path,h,w):
    lab,_c,nm=nib.freesurfer.io.read_annot(path.format(h=h)); nm=[x.decode() if isinstance(x,bytes) else x for x in nm]
    idx=[i for i,x in enumerate(nm) if x in w]; return np.isin(lab,idx)
def shade(coords,F,light=np.array([-1.0,-0.3,0.6])):
    t=coords[F]; n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]); n/=(np.linalg.norm(n,axis=1,keepdims=True)+1e-9)
    light=light/np.linalg.norm(light); return 0.55+0.45*np.clip(np.abs(n@light),0,1)

def render(ax,h,azim):
    coords,faces=nib.freesurfer.read_geometry(SURF.format(h=h))
    lab=np.asarray(nib.freesurfer.io.read_annot(ANN.format(h=h,v=VER))[0])
    ins=sub(DEST,h,INS); fjk=sub(ECON,h,["FJK"])
    c=coords[ins]; lo=c.min(0)-20; hi=c.max(0)+20
    inv=np.all((coords>=lo)&(coords<=hi),axis=1); F=faces[inv[faces].all(axis=1)]
    sh=shade(coords,F)
    fmaj=np.array([np.bincount(lab[r][lab[r]>0]).argmax() if (lab[r]>0).any() else 0 for r in F])
    rgb=np.clip(np.array([CYTO7.get(int(t),(0.88,0.88,0.88)) for t in fmaj])*sh[:,None],0,1)
    ax.add_collection3d(Poly3DCollection(coords[F],facecolors=rgb,edgecolors="none",linewidths=0,shade=False))
    ffjk=fjk[F].sum(1)>=2
    if ffjk.any():
        ax.add_collection3d(Poly3DCollection(coords[F[ffjk]],facecolors=(1,0,1,0.0),
                            edgecolors=(1.0,0.0,1.0,0.55),linewidths=0.35))
    for s,a in ((ax.set_xlim,0),(ax.set_ylim,1),(ax.set_zlim,2)): s(lo[a],hi[a])
    ax.set_box_aspect(hi-lo); ax.view_init(elev=0,azim=azim); ax.set_axis_off()
    xm=coords[ins][:,0].mean(); z0=lo[2]+2
    ax.text(xm,hi[1]-3,z0,"anterior",fontsize=9,style="italic",color="0.25")
    ax.text(xm,lo[1]+3,z0,"posterior",fontsize=9,style="italic",color="0.25")
    agr_ins=int(((lab==2)&ins).sum()); agr_fjk=int(((lab==2)&fjk).sum())
    return agr_ins,int(ins.sum()),agr_fjk,int(fjk.sum())

fig=plt.figure(figsize=(15,7)); fig.patch.set_facecolor("white"); info={}
for i,(h,azim) in enumerate((("lh",180),("rh",0))):
    ax=fig.add_subplot(1,2,i+1,projection="3d"); info[h]=render(ax,h,azim)
    ai,it,af,ft=info[h]
    ax.set_title(f"{h.upper()} insula ({VER}) — agranular {ai}/{it} ({100*ai/it:.1f}%);  FJK filled {af}/{ft} ({100*af/ft:.0f}%)",
                 fontsize=10.5,weight="bold")
handles=[Patch(facecolor=CYTO7[c],label=NAMES[c]) for c in range(1,8)]
handles.append(Line2D([0],[0],color=(1,0,1),lw=1.5,label="von Economo FJK/FI (agranular target)"))
fig.legend(handles=handles,loc="lower center",ncol=4,fontsize=9,frameon=False,bbox_to_anchor=(0.5,0.0))
fig.suptitle(f"cyto7 {VER} insula (inflated, lateral): painted AGRANULAR (blue) vs von-Economo agranular target (FJK outline)",
             fontsize=13,weight="bold",y=0.99)
fig.subplots_adjust(bottom=0.13,top=0.9,wspace=0.02)
os.makedirs(os.path.dirname(OUT),exist_ok=True)
fig.savefig(OUT,dpi=155,facecolor="white",bbox_inches="tight")
print("saved",OUT)
