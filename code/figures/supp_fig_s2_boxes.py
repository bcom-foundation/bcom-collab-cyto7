import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg
ROOT="."
CACHE=f"{ROOT}/resources/cyto7_derived/cache"; NM=f"{ROOT}/resources/neuromaps_cache"
lab={H:np.load(f"{CACHE}/v9_labels_fsLR32k_hemi-{H}.npy") for H in "LR"}
def load(pre): return {H:np.load(f"{NM}/{pre}_fsLR32k_hemi-{H}.npy") for H in "LR"}
BANDS=["megdelta","megtheta","megalpha","megbeta","meggamma1"]; FB=np.array([2.25,6.0,10.0,21.0,45.0])
band={b:load("hcps1200_"+b) for b in BANDS}; ts=load("hcps1200_megtimescale")
# spectral centroid per vertex per hemi
cent={}
for H in "LR":
    P=np.stack([band[b][H] for b in BANDS],1); psum=P.sum(1)
    with np.errstate(invalid="ignore",divide="ignore"): C=(P*FB).sum(1)/psum
    C[~(psum>0)|~np.isfinite(P).all(1)]=np.nan; cent[H]=C
alllab=np.concatenate([lab["L"],lab["R"]])
allcent=np.concatenate([cent["L"],cent["R"]]); allts=np.concatenate([ts["L"],ts["R"]])
NAMES={2:"Agr",3:"Dys",4:"EulI",5:"EulII",6:"EulIII",7:"Konio"}; types=list(range(2,8))
# released stats (allo-excluded, clean/v9 lineage; from figures/REPORT.md Sec.7)
STAT={"cent":dict(rho=+0.46,p=0.068,q=0.068),"ts":dict(rho=-0.51,p=0.026,q=0.068)}
PANELS=[("cent",allcent,"Spectral centroid (Hz)","power-weighted mean frequency"),
        ("ts",allts,"Intrinsic timescale (a.u.)","megtimescale")]
def boxvals(vals):
    out=[]
    for t in types:
        sel=(alllab==t)&np.isfinite(vals); out.append(vals[sel])
    return out
w_in=190/25.4; fig,axes=plt.subplots(1,2,figsize=(w_in,w_in*0.42)); fig.patch.set_facecolor("white")
tcol=[plt.cm.viridis((t-1)/6.0) for t in types]
for ax,(key,vals,ylab,sub) in zip(axes,PANELS):
    data=boxvals(vals)
    bp=ax.boxplot(data,positions=range(len(types)),widths=0.62,showfliers=False,patch_artist=True,
                  medianprops=dict(color="black",lw=1.0),
                  whiskerprops=dict(color="0.4",lw=0.8),capprops=dict(color="0.4",lw=0.8),
                  boxprops=dict(lw=0.6))
    for patch,c in zip(bp["boxes"],tcol): patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("0.25")
    med=[np.median(d) if len(d) else np.nan for d in data]
    ax.plot(range(len(types)),med,color="#c0392b",lw=1.4,marker="o",ms=3.2,zorder=5)
    s=STAT[key]; sig="*" if s["q"]<0.05 else ""
    ax.set_xticks(range(len(types))); ax.set_xticklabels([NAMES[t] for t in types],rotation=30,ha="right",fontsize=7.5)
    ax.tick_params(axis="y",labelsize=7.5)
    ax.set_ylabel(ylab,fontsize=8.5); ax.set_title(f"{ylab.split(' (')[0]} ({sub})",fontsize=8.5,fontweight="bold")
    ax.grid(axis="y",ls="--",alpha=0.35)
    ax.text(0.03,0.03,f"$\\rho$={s['rho']:+.2f}  spin $p$={s['p']:.3f}  $q$={s['q']:.3f}{sig}",
            transform=ax.transAxes,fontsize=7.5,va="bottom",ha="left",
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="0.7",lw=0.6))
fig.tight_layout()
OUT=str(cfg.figures_dir()/"cyto7_supp_frequency_timescale.png")
cfg.figures_dir().mkdir(parents=True,exist_ok=True)
fig.savefig(OUT,dpi=600,facecolor="white"); print("saved",OUT)
from PIL import Image; im=Image.open(OUT); print("px",im.size,"->",round(im.size[0]/600*25.4,1),"x",round(im.size[1]/600*25.4,1),"mm")
