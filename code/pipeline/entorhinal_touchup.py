import os, numpy as np, nibabel as nib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg
# This is a v8 diagnostic kept for provenance: it renders the entorhinal region that the
# v8-to-v9 even-up changed. The surface geometry and the von Economo aparc are third-party
# inputs and are not redistributed here, so it needs CYTO7_DATA_DIR set; the v8 annot is
# located in the released atlas tree.
ROOT=str(cfg.REPO_ROOT)
SURF=str(cfg.data_dir("fsaverage_surfaces")/"{h}.inflated")
ANN=str(cfg.ATLAS/"provenance"/"versions"/"pial.{h}.cyto7.v8.annot")
APARC=str(cfg.data_dir("voneconomo")/"{h}.aparc.annot")
OUT=str(cfg.figures_dir()/"entorhinal_LR_touchup_v8.png")
PAL={0:(0.86,0.92,0.98),1:(0.55,0.20,0.65),2:(0.20,0.35,0.85),3:(0.10,0.62,0.70),4:(0.25,0.70,0.35),5:(0.62,0.78,0.20),6:(0.98,0.62,0.10),7:(0.98,0.85,0.15)}
NM={1:"Allocortex",2:"Agranular",3:"Dysgranular",4:"Eulaminate I",5:"Eulaminate II",6:"Eulaminate III",7:"Koniocortex"}
REGION=["entorhinal","temporalpole","parahippocampal"]
def amask(h,wanted):
    lab,_c,names=nib.freesurfer.io.read_annot(APARC.format(h=h)); names=[x.decode() if isinstance(x,bytes) else x for x in names]
    idx=[i for i,x in enumerate(names) if x in wanted]; return np.isin(lab,idx)
def shade(coords,F,light=np.array([-0.3,-0.3,-1.0])):
    t=coords[F]; n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]); n/=np.linalg.norm(n,axis=1,keepdims=True)+1e-9
    light=light/np.linalg.norm(light); return 0.6+0.4*np.clip(np.abs(n@light),0,1)
def render(ax,h):
    coords,faces=nib.freesurfer.read_geometry(SURF.format(h=h)); lab=np.asarray(nib.freesurfer.io.read_annot(ANN.format(h=h))[0])
    reg=amask(h,REGION); ento=amask(h,["entorhinal"])
    c=coords[reg]; lo=c.min(0)-14; hi=c.max(0)+14
    inv=np.all((coords>=lo)&(coords<=hi),axis=1); F=faces[inv[faces].all(axis=1)]
    sh=shade(coords,F); fl=lab[F]
    fmaj=np.array([np.bincount(r[r>0],minlength=8).argmax() if (r>0).any() else 0 for r in fl])
    rgb=np.clip(np.array([PAL.get(int(t),(0.85,0.85,0.85)) for t in fmaj])*sh[:,None],0,1)
    # target (LH only): entorhinal & currently agranular -> convert to allocortex
    tgt=ento & (lab==2)
    ft=tgt[F].sum(1)>=2
    if h=="lh" and ft.any():
        rgb[ft]=np.clip(np.array([0.90,0.05,0.05])*sh[ft,None]*0.6+rgb[ft]*0.4,0,1)
    ax.add_collection3d(Poly3DCollection(coords[F],facecolors=rgb,edgecolors="none",linewidths=0,shade=False))
    if h=="lh" and ft.any():
        ax.add_collection3d(Poly3DCollection(coords[F[ft]],facecolors=(0.9,0,0,0.0),edgecolors=(0.85,0,0,0.9),linewidths=0.3))
    # outline the entorhinal parcel (both) in cyan so extent is comparable
    fe=ento[F].sum(1)>=2
    if fe.any():
        ax.add_collection3d(Poly3DCollection(coords[F[fe]],facecolors=(0,0,0,0.0),edgecolors=(0.0,0.75,0.85,0.7),linewidths=0.3))
    for s,a in ((ax.set_xlim,0),(ax.set_ylim,1),(ax.set_zlim,2)): s(lo[a],hi[a])
    ax.set_box_aspect(hi-lo); ax.view_init(elev=-88,azim=0); ax.set_axis_off()
    xm=coords[reg][:,0].mean()
    ax.text(xm,hi[1]-2,lo[2]+2,"anterior",fontsize=9,style="italic",color="0.3")
    ax.text(xm,lo[1]+2,lo[2]+2,"posterior",fontsize=9,style="italic",color="0.3")
    ea=int((ento&(lab==2)).sum()); et=int(ento.sum()); eallo=int((ento&(lab==1)).sum())
    return ea,et,eallo
fig=plt.figure(figsize=(15,7.5)); fig.patch.set_facecolor("white"); info={}
for i,h in enumerate(("lh","rh")):
    ax=fig.add_subplot(1,2,i+1,projection="3d"); info[h]=render(ax,h)
    ea,et,eallo=info[h]
    if h=="lh":
        ax.set_title(f"LH entorhinal — allo {100*eallo/et:.0f}%, agr {100*ea/et:.0f}%\nconvert {ea} agranular → allocortex",fontsize=11,weight="bold",pad=2)
    else:
        ax.set_title(f"RH entorhinal — allo {100*eallo/et:.0f}%, agr {100*ea/et:.0f}%\nreference (already allocortex)",fontsize=11,weight="bold",pad=2)
handles=[Patch(facecolor=PAL[c],label=NM[c]) for c in (1,2,3,4)]
handles+=[Line2D([0],[0],color=(0.0,0.75,0.85),lw=2,label="Desikan entorhinal parcel"),
          Patch(facecolor=(0.85,0.1,0.1),label="LH: agranular → paint ALLOCORTEX (match RH)")]
fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=9,frameon=False,bbox_to_anchor=(0.5,0.0))
lla=100*info["lh"][2]/info["lh"][1]; rra=100*info["rh"][2]/info["rh"][1]
fig.suptitle(f"v8 L/R touch-up — even up the entorhinal allocortex (LH {lla:.0f}% vs RH {rra:.0f}%; ventral view)",fontsize=13,weight="bold",y=0.99)
fig.text(0.5,-0.04,"Paint the red LH entorhinal vertices (currently agranular) as ALLOCORTEX to match the right hemisphere; the topology repair then settles the agranular→dysgranular buffer. Temporal-pole edge (RH 15% allo, LH 0%) is a smaller, optional match.",ha="center",fontsize=8.5,style="italic",color="0.35",wrap=True)
fig.subplots_adjust(bottom=0.13,top=0.9,wspace=0.02)
fig.savefig(OUT,dpi=160,facecolor="white",bbox_inches="tight"); print("saved",OUT)
for h,(ea,et,eallo) in info.items(): print(f"{h}: entorhinal n={et}, allocortex={eallo} ({100*eallo/et:.0f}%), agranular={ea} ({100*ea/et:.0f}%)")
