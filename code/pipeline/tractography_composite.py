import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import os;OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"figures","v9","manuscript","figure_7_tractography_composite.png")
SH=pd.read_csv("figures/v9/tractography/connectivity_short.csv",index_col=0)
LO=pd.read_csv("figures/v9/tractography/connectivity_long.csv",index_col=0)
nul=pd.read_csv("figures/v9/tractography/tractography_null.csv").iloc[0]
LAB=["Allo","Agr","Dys","EulI","EulII","EulIII","Konio"]
vmax=max(SH.values.max(),LO.values.max()); norm=LogNorm(vmin=1,vmax=vmax); cmap=plt.get_cmap("magma")
fig=plt.figure(figsize=(17,5.4)); fig.patch.set_facecolor("white")
gs=fig.add_gridspec(1,5,width_ratios=[1,1,0.06,0.32,1.35],wspace=0.28)
def mat(ax,M,title):
    A=M.values.astype(float); im=ax.imshow(np.where(A>0,A,np.nan),cmap=cmap,norm=norm)
    ax.set_xticks(range(7)); ax.set_xticklabels(LAB,rotation=40,ha="right",fontsize=8)
    ax.set_yticks(range(7)); ax.set_yticklabels(LAB,fontsize=8)
    for i in range(7):
        for j in range(7):
            v=A[i,j]
            if v>0: ax.text(j,i,f"{int(v)}",ha="center",va="center",fontsize=6.5,color="white" if v<0.15*vmax else "0.1")
    ax.set_title(title,fontsize=11,weight="bold")
    ax.set_xticks(np.arange(-.5,7,1),minor=True); ax.set_yticks(np.arange(-.5,7,1),minor=True)
    ax.grid(which="minor",color="0.9",lw=0.5); ax.tick_params(which="minor",length=0)
    return im
axs=fig.add_subplot(gs[0,0]); im=mat(axs,SH,"Short-range (<80 mm)")
axl=fig.add_subplot(gs[0,1]); mat(axl,LO,"Long-range (>80 mm)")
cax=fig.add_subplot(gs[0,2]); cb=fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax); cb.set_label("streamline count (log)",fontsize=9)
# scatter: type-distance vs short count (off-diagonal)
axc=fig.add_subplot(gs[0,4])
A=SH.values; xs=[]; ys=[]
for i in range(7):
    for j in range(i+1,7):
        xs.append(j-i); ys.append(A[i,j])
xs=np.array(xs,float); ys=np.array(ys,float)
axc.scatter(xs+np.random.default_rng(0).uniform(-0.06,0.06,len(xs)),ys,s=55,color="#5b4b8a",alpha=0.75,edgecolor="w",lw=0.5)
xx=np.linspace(1,6,100); b,a=np.polyfit(xs,np.log1p(ys),1); axc.plot(xx,np.expm1(a+b*xx),color="#b2182b",lw=2,
    label=f"log-linear fit (slope={nul.short_slope_obs:+.2f})")
axc.set_xlabel("Type-distance |i − j| (steps along the gradient)",fontsize=10)
axc.set_ylabel("Short-range streamline count",fontsize=10)
axc.set_title("Connectivity decays with cytoarchitectural type-distance",fontsize=11,weight="bold")
surv="survives" if nul.partial_p<0.05 else "does not survive"
axc.text(0.97,0.72,f"perm-null p = {nul.slope_perm_p:.3f}\npartial r (| contact area) = {nul.partial_r_typedist_given_contact:+.2f}\n   p = {nul.partial_p:.3f} ({surv})",
         transform=axc.transAxes,ha="right",va="top",fontsize=8.5,bbox=dict(boxstyle="round,pad=0.3",fc="0.96",ec="0.7"))
axc.legend(fontsize=9,loc="upper right"); axc.grid(alpha=0.3,ls=":")
fig.suptitle("Tractography: short- vs long-range type-by-type connectivity and the type-distance falloff",fontsize=13,weight="bold",y=1.0)
fig.savefig(OUT,dpi=170,facecolor="white",bbox_inches="tight"); print("saved; vmax",int(vmax),"slope",nul.short_slope_obs)
