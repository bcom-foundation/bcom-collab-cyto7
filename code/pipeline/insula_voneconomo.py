"""Insula in the von Economo-Koskinas parcellation (inflated, lateral view).

Row A: von Economo AREAS in/around the insula (qualitative, labelled by acronym).
Row B: the same region coloured by the Garcia-Cabezas-derived cyto7 TYPE, directly
       comparable to insula_paint_guide_inflated.png.

Purpose: show that von Economo places an AGRANULAR sector (FJK = frontoinsular FI +
frontal piriform) at the ventral-anterior insula, while the insula proper is IA
(precentral, dysgranular) and IB (postcentral, eulaminate I) -- i.e. the GC-consistent
agranular target is the FJK tip, not the whole anterior insula.

Usage:  python scripts/insula_voneconomo.py   (nibabel + matplotlib only)
"""
import os, csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURF = os.path.join(ROOT, "resources", "fsaverage_surfaces", "{h}.inflated")
ECON = os.path.join(ROOT, "resources", "voneconomo", "{h}.economo.annot")
DEST = os.path.join(ROOT, "resources", "refine_atlases", "{h}.aparc.a2009s.annot")
TYPES = os.path.join(ROOT, "resources", "voneconomo", "von_economo_cortical_types.csv")
OUT = os.path.join(ROOT, "figures", "v9", "adjudication", "insula_voneconomo.png")

CYTO7 = {1:(0.55,0.20,0.65),2:(0.20,0.35,0.85),3:(0.10,0.62,0.70),4:(0.25,0.70,0.35),
         5:(0.62,0.78,0.20),6:(0.98,0.62,0.10),7:(0.98,0.85,0.15)}
CNAME = {0:"periallo/limbic",1:"Allocortex",2:"Agranular",3:"Dysgranular",4:"Eulaminate I",
         5:"Eulaminate II",6:"Eulaminate III",7:"Koniocortex"}
INS = ["G_Ins_lg_and_S_cent_ins","G_insular_short","S_circular_insula_ant",
       "S_circular_insula_inf","S_circular_insula_sup"]

A2C, A2N = {}, {}
with open(TYPES, newline="") as f:
    for r in csv.DictReader(f):
        c = r["cyto7_code"].strip()
        A2C[r["economo_acronym"]] = int(c) if c else 0
        A2N[r["economo_acronym"]] = r["economo_area_name"]
QUAL = plt.get_cmap("tab20")


def dm(h, w):
    lab,_c,names = nib.freesurfer.io.read_annot(DEST.format(h=h))
    names=[n.decode() if isinstance(n,bytes) else n for n in names]
    idx=[i for i,nm in enumerate(names) if nm in w]; return np.isin(lab,idx)


def shade(coords, F, light=np.array([-1.0,-0.3,0.6])):
    t=coords[F]; n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    n/=(np.linalg.norm(n,axis=1,keepdims=True)+1e-9); light=light/np.linalg.norm(light)
    return 0.55+0.45*np.clip(np.abs(n@light),0,1)


def facemaj(vals, F):
    fl=vals[F]; out=np.zeros(len(F),int)
    for i in range(len(F)):
        v=fl[i]; v=v[v>0]
        if len(v): out[i]=np.bincount(v).argmax()
    return out


def load(hemi):
    coords,faces=nib.freesurfer.read_geometry(SURF.format(h=hemi))
    elab,_c,en=nib.freesurfer.io.read_annot(ECON.format(h=hemi))
    en=[n.decode() if isinstance(n,bytes) else n for n in en]
    ins=dm(hemi,INS); c=coords[ins]; lo=c.min(0)-20; hi=c.max(0)+20
    inv=np.all((coords>=lo)&(coords<=hi),axis=1); F=faces[inv[faces].all(axis=1)]
    return coords,F,np.asarray(elab),en,lo,hi,ins


def panel(ax,coords,F,facecol,lo,hi,azim,ins):
    ax.add_collection3d(Poly3DCollection(coords[F],facecolors=facecol,edgecolors="none",linewidths=0,shade=False))
    for s,a in ((ax.set_xlim,0),(ax.set_ylim,1),(ax.set_zlim,2)): s(lo[a],hi[a])
    ax.set_box_aspect(hi-lo); ax.view_init(elev=0,azim=azim); ax.set_axis_off()
    xm=coords[ins][:,0].mean(); z0=lo[2]+2
    ax.text(xm,hi[1]-3,z0,"anterior",fontsize=9,style="italic",color="0.25")
    ax.text(xm,lo[1]+3,z0,"posterior",fontsize=9,style="italic",color="0.25")


def main(which="both", dpi=150):
    data={}; present=[]
    for hemi,azim in (("lh",180),("rh",0)):
        coords,F,elab,en,lo,hi,ins=load(hemi)
        fa=facemaj(elab,F)
        data[hemi]=(coords,F,en,fa,lo,hi,azim,ins)
        for fid in np.unique(fa):
            if fid>0 and en[fid] not in present: present.append(en[fid])
    present=sorted(present, key=lambda a:(A2C.get(a,9),a))
    acol={a:QUAL(i%20)[:3] for i,a in enumerate(present)}
    os.makedirs(os.path.dirname(OUT),exist_ok=True)

    def render_fig(mode):
        fig=plt.figure(figsize=(15,6.5)); fig.patch.set_facecolor("white")
        for col,hemi in enumerate(("lh","rh")):
            coords,F,en,fa,lo,hi,azim,ins=data[hemi]; sh=shade(coords,F)
            ax=fig.add_subplot(1,2,col+1,projection="3d")
            if mode=="areas":
                rgb=np.array([acol.get(en[fid],(0.9,0.9,0.9)) if fid>0 else (0.9,0.9,0.9) for fid in fa])
                ax.set_title(f"{hemi.upper()} — von Economo areas",fontsize=12,weight="bold")
            else:
                ft=np.array([A2C.get(en[fid],0) if fid>0 else 0 for fid in fa])
                rgb=np.array([CYTO7.get(int(t),(0.88,0.88,0.88)) for t in ft])
                ax.set_title(f"{hemi.upper()} — von Economo → cyto7 type",fontsize=12,weight="bold")
            panel(ax,coords,F,np.clip(rgb*sh[:,None],0,1),lo,hi,azim,ins)
        if mode=="areas":
            leg=[Patch(facecolor=acol[a],label=f"{a}: {A2N.get(a,'')[:26]} ({CNAME[A2C.get(a,0)]})") for a in present]
            fig.legend(handles=leg,loc="center left",bbox_to_anchor=(0.88,0.5),fontsize=6.5,
                       frameon=False,title="von Economo areas (near insula)")
            sub="areas"
        else:
            leg=[Patch(facecolor=CYTO7[c],label=CNAME[c]) for c in range(1,8)]
            leg.append(Patch(facecolor=(0.88,0.88,0.88),label=CNAME[0]))
            fig.legend(handles=leg,loc="center left",bbox_to_anchor=(0.88,0.5),fontsize=8,
                       frameon=False,title="cyto7 type (GC mapping)")
            sub="types"
        fig.suptitle("Insula in von Economo-Koskinas (inflated, lateral): FJK = frontoinsular AGRANULAR tip; "
                     "IA precentral = dysgranular; IB postcentral = eulaminate I",fontsize=11.5,weight="bold",y=0.99)
        fig.subplots_adjust(left=0.01,right=0.87,top=0.9,bottom=0.02,wspace=0.02)
        p=OUT.replace(".png",f"_{sub}.png")
        fig.savefig(p,dpi=dpi,facecolor="white",bbox_inches="tight"); plt.close(fig)
        print("saved",p)

    if which in ("areas","both"): render_fig("areas")
    if which in ("types","both"): render_fig("types")


if __name__=="__main__":
    import sys
    which = sys.argv[1] if len(sys.argv)>1 else "both"
    main(which)
