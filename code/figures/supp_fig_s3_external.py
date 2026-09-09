import time; t0=time.time()
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg
import numpy as np, nibabel as nib, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib import cm; from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from scipy import stats as ss
from cyto7_surface_io import surface_path
ROOT="."; CACHE=ROOT+"/resources/cyto7_derived/cache"; NM=ROOT+"/resources/neuromaps_cache"
# Named for the file this script actually produces in the release, which is the
# supplementary external-validation figure.
OUT=str(cfg.figures_dir()/"cyto7_supp_external_validation.png")
NAMES={1:"Allo",2:"Agr",3:"Dys",4:"EulI",5:"EulII",6:"EulIII",7:"Konio"}
tab=pd.read_csv([p for p in __import__('glob').glob('figures/**/external_validation_table.csv',recursive=True)][0]).set_index("FeatureKey")
lab={H:np.load(f"{CACHE}/v9_labels_fsLR32k_hemi-{H}.npy") for H in "LR"}
geo={H:(lambda g:(np.asarray(g.darrays[0].data,float),np.asarray(g.darrays[1].data,int)))(nib.load(str(surface_path("Validation210",H,"inflated")))) for H in "LR"}
def load(pre): return {H:np.load(f"{NM}/{pre}_fsLR32k_hemi-{H}.npy") for H in "LR"}
gene=load("abagen_genepc1"); rec=load("hansen_receptorpc1")
prof={H:np.load(f"{NM}/bigbrain_profiles_fsLR32k_hemi-{H}.npy") for H in "LR"}
bbskew={H:ss.skew(prof[H],axis=0,nan_policy="omit") for H in "LR"}
FEATURES=[("AHBA gene expression PC1","a.u.",gene,"genepc1"),
          ("PET receptor PC1","a.u.",rec,"receptorpc1"),
          ("BigBrain profile skewness","a.u.",bbskew,"bigbrain_profile_skewness")]
VIEWS=[("L","lateral"),("L","medial"),("R","lateral"),("R","medial")]  # 2x2: LH top, RH bottom
def proj(H,view,c):
    y,z=c[:,1],c[:,2]; return (-y,z) if (H,view) in (("L","lateral"),("R","medial")) else (y,z)
G={}
for H,view in VIEWS:
    coords,faces=geo[H]; lb=lab[H]
    n=np.cross(coords[faces][:,1]-coords[faces][:,0],coords[faces][:,2]-coords[faces][:,0]); n/=np.linalg.norm(n,axis=1,keepdims=True)+1e-9
    fr=(n[:,0]<0) if (H,view) in (("L","lateral"),("R","medial")) else (n[:,0]>0)
    F=faces[fr]; fx=coords[F].mean(1)[:,0]
    order=np.argsort(-fx) if (H,view) in (("L","lateral"),("R","medial")) else np.argsort(fx); F=F[order]
    sx,sy=proj(H,view,coords); verts=np.stack([sx[F],sy[F]],axis=-1)
    sh=0.62+0.38*np.clip(np.abs(n[fr,2][order]),0,1); medf=(lb[F]==0).sum(1)>=2
    E=np.unique(np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[0,2]]]),axis=1),axis=0)
    frv=np.zeros(len(coords),bool); frv[F.ravel()]=True
    bd=(lb[E[:,0]]!=lb[E[:,1]])&(lb[E[:,0]]>0)&(lb[E[:,1]]>0)&frv[E[:,0]]&frv[E[:,1]]
    segs=np.stack([np.stack([sx[E[bd,0]],sy[E[bd,0]]],1),np.stack([sx[E[bd,1]],sy[E[bd,1]]],1)],1)
    G[(H,view)]=dict(F=F,verts=verts,sh=sh,medf=medf,segs=segs,lim=(sx.min(),sx.max(),sy.min(),sy.max()))
nrow=len(FEATURES); cmap=plt.get_cmap("magma")
# 190 mm wide @ 600 dpi, no title / no caption line; compact vertical packing so the caption
# fits on the page (fonts scaled by fs for the reduced canvas).
W_MM=190.0; w_in=W_MM/25.4; h_in=8.1; fs=0.82
fig=plt.figure(figsize=(w_in,h_in)); fig.patch.set_facecolor("white")
outer=fig.add_gridspec(nrow+1,4,width_ratios=[2.0,0.06,0.55,1.45],height_ratios=[1]*nrow+[0.72],
                       wspace=0.06,hspace=0.40,left=0.065,right=0.965,top=0.955,bottom=0.095)
for r,(name,unit,d,key) in enumerate(FEATURES):
    allv=np.concatenate([d[H][lab[H]>0] for H in "LR"]); allv=allv[np.isfinite(allv)]
    vmin,vmax=np.percentile(allv,2),np.percentile(allv,98); norm=Normalize(vmin,vmax)
    sub=outer[r,0].subgridspec(2,2,wspace=0.02,hspace=0.08)
    for i,(H,view) in enumerate(VIEWS):
        ax=fig.add_subplot(sub[i//2,i%2]); g=G[(H,view)]; val=d[H]
        fv=np.nanmean(val[g["F"]],axis=1); col=cmap(norm(fv))[:,:3]
        bad=g["medf"]|~np.isfinite(fv); col[bad]=(0.68,0.85,0.90); col=np.clip(col*g["sh"][:,None],0,1)
        ax.add_collection(PolyCollection(g["verts"],facecolors=col,edgecolors="face",linewidths=0.3))
        ax.add_collection(LineCollection(g["segs"],colors="white",linewidths=0.25,alpha=0.55))
        x0,x1,y0,y1=g["lim"]; ax.set_xlim(x0-2,x1+2); ax.set_ylim(y0-2,y1+2); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{H}H {view}",fontsize=9*fs,pad=1.5)
    cax=fig.add_subplot(outer[r,1]); cb=fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax)
    cb.set_label(f"{name}",fontsize=9*fs); cb.ax.tick_params(labelsize=8*fs)
    axb=fig.add_subplot(outer[r,3])
    for t in range(1,8):
        for H,off,fc,mc in (("L",-0.18,"0.35","w"),("R",0.18,"0.78","0.2")):
            v=d[H][(lab[H]==t)&np.isfinite(d[H])]
            if len(v): axb.boxplot(v,positions=[t+off],widths=0.32,showfliers=False,patch_artist=True,boxprops=dict(facecolor=fc,edgecolor="0.3"),medianprops=dict(color=mc),whiskerprops=dict(color=fc),capprops=dict(color=fc))
    s=tab.loc[key]; sig="*" if s.p_spin_fdr<0.05 else ""
    axb.set_xticks(range(1,8))
    if r==nrow-1:
        axb.set_xticklabels([NAMES[t] for t in range(1,8)],rotation=30,ha="right",fontsize=8*fs)
    else:
        axb.set_xticklabels([])
    axb.tick_params(axis="y",labelsize=8*fs)
    axb.set_ylabel(unit,fontsize=9*fs); axb.set_xlim(0.4,7.6); axb.grid(axis="y",ls=":",alpha=0.4)
    axb.set_title(f"{name}\n(Spearman $\\rho$={s.spearman_rho:+.2f}, spin p={s.p_spin:.3f}, q={s.p_spin_fdr:.3f}{sig})",fontsize=9.5*fs)
    if r==0: axb.legend(handles=[Patch(facecolor="0.35",label="LH"),Patch(facecolor="0.78",label="RH")],loc="upper right",fontsize=8*fs,frameon=False)
# --- BigBrain mean intensity profile by cyto7 type (staining vs cortical depth) ---
allprof=np.concatenate([prof["L"],prof["R"]],axis=1); alllab=np.concatenate([lab["L"],lab["R"]])
depth=np.linspace(0,1,allprof.shape[0])
subp=outer[nrow,:].subgridspec(1,3,width_ratios=[0.85,2.3,0.85]); axp=fig.add_subplot(subp[0,1])
for t in range(1,8):
    sel=(alllab==t)&np.isfinite(allprof).all(0)
    if sel.sum(): axp.plot(depth,np.nanmean(allprof[:,sel],axis=1),color=plt.cm.viridis((t-1)/6.0),lw=1.6,label=NAMES[t])
axp.set_xlabel("cortical depth (pial → white matter)",fontsize=10*fs); axp.set_ylabel("BigBrain staining intensity (a.u.)",fontsize=10*fs)
axp.tick_params(labelsize=8*fs)
axp.set_title("BigBrain mean intensity profile by cyto7 type",fontsize=11*fs,weight="bold")
axp.grid(alpha=0.3,ls=":"); axp.legend(ncol=7,fontsize=8*fs,frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.32),columnspacing=1.0,handlelength=1.4)
fig.savefig(OUT,dpi=600,facecolor="white")   # exact figsize -> 190 mm wide @ 600 dpi
print("saved in",round(time.time()-t0,1),"s")
