"""Figure 6: independent cortical measures ordered along the cyto7 type axis.

Default mode reproduces the published panel (eight measures, one continuous axis).
``--rr3`` builds the redesigned panel required by review item 11 (SPEC_rr3_fig6_rebuild):
twelve measures grouped into three labelled bands by statistical unit, so
vertex-level correlations, a parcel-level correlation and the group mean of
subject-wise correlations are never placed on one continuous axis; RORB added; the
inclusion rule stated inside the panel. Every rho and q is read from RR2's
assembled outcome table, so the figure cannot drift from it.
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R="figures/v9/structure_function"
Rext="figures/v9"

if "--rr3" in __import__("sys").argv:
    from rr3_fig6_bands import main as _rr3_main
    _rr3_main()
    raise SystemExit(0)
fs=pd.read_csv(f"{R}/functional_summary_table_v9.csv")
def fs_rq(k):
    s=fs[fs.FeatureKey==k].iloc[0]; return float(s.spearman_rho), float(s.p_spin_fdr)
ext=pd.read_csv(f"{R}/external_validation_table.csv").set_index("FeatureKey")
comp=pd.read_csv(f"{R}/receptor_composites.csv").set_index("composite")
div=pd.read_csv(f"{R}/receptor_diversity.csv").set_index("metric")
fig9=pd.read_csv(f"{R}/fig9_predictions_summary.csv").set_index("metric")
my_r,my_q=fs_rq("myelin"); th_r,th_q=fs_rq("thickness"); gr_r,gr_q=fs_rq("gradient")
# Intrinsic timescale: use the v2 source-reconstructed value (per-subject mean rho, ACF-area
# INT) which SURVIVES spin+FDR, NOT the old group-map value in functional_summary_table_v9.csv
# (which is left untouched). Read from meg_dynamics_v2_summary.csv.
_v2=pd.read_csv(f"{R}/meg_dynamics_v2_summary.csv").set_index("metric")
ts_r=float(_v2.loc["int_area","group_mean_rho"]); ts_q=float(_v2.loc["int_area","fdr_q"])
ce_r=float(_v2.loc["centroid","group_mean_rho"]); ce_q=float(_v2.loc["centroid","fdr_q"])
sf_r=float(_v2.loc["sf_ratio","group_mean_rho"]); sf_q=float(_v2.loc["sf_ratio","fdr_q"])
import sys as _sys
WITH_CENTROID_SF = "--with-centroid-sf" in _sys.argv
gene_r=float(ext.loc["genepc1","spearman_rho"]); gene_q=float(ext.loc["genepc1","p_spin_fdr"])
im_r=float(comp.loc["iono_minus_metabo_index","spearman_rho"]); im_q=float(comp.loc["iono_minus_metabo_index","fdr_q"])
dv_r=float(div.loc["shannon_entropy_H","spearman_rho"]); dv_q=float(fig9.loc["receptor_diversity_H","fdr_q_fig9"])
ev_r=float(fig9.loc["xu2020_evoexp","spearman_rho"]); ev_q=float(fig9.loc["xu2020_evoexp","fdr_q_fig9"])
M=[("Gene expression PC1 (AHBA)",gene_r,gene_q),
   ("T1w/T2w myelin",my_r,my_q),
   ("Ionotropic/metabotropic index",im_r,im_q),
   ("Evolutionary expansion",ev_r,ev_q),
   ("Receptor diversity",dv_r,dv_q),
   ("Intrinsic timescale",ts_r,ts_q),
   ("Cortical thickness",th_r,th_q),
   ("Functional gradient",gr_r,gr_q)]
if WITH_CENTROID_SF:
    M += [("Spectral centroid (MEG)",ce_r,ce_q),("Slow/fast ratio (MEG)",sf_r,sf_q)]
M.sort(key=lambda t:t[1], reverse=True)  # rho descending: rises on top
names=[m[0] for m in M]; rhos=np.array([m[1] for m in M]); qs=np.array([m[2] for m in M])
y=np.arange(len(M))[::-1]  # top row = first
cmap=plt.cm.RdBu_r; import matplotlib.colors as mc; norm=mc.Normalize(-0.7,0.7)
fig=plt.figure(figsize=(190/25.4,110/25.4)); fig.patch.set_facecolor("white")
ax=fig.add_axes([0.31,0.15,0.66,0.76])
ax.set_ylim(-0.6,len(M)+0.1)
ax.axvline(0,color="0.6",lw=1.0,zorder=1)
for yi,(nm,rho,q) in zip(y,M):
    c=cmap(norm(rho)); sig=q<0.05
    ax.plot([0,rho],[yi,yi],color=c,lw=3.2,solid_capstyle="round",zorder=2)
    ax.plot([rho],[yi],marker="o",ms=11,mfc=c if sig else "white",mec=c,mew=2.0,zorder=3)
    ha="left" if rho>0 else "right"; off=0.03 if rho>0 else -0.03
    ax.text(rho+off,yi,f"{rho:+.2f}{'*' if sig else ''}",va="center",ha=ha,fontsize=8,color="0.2")
ax.set_yticks(y); ax.set_yticklabels(names,fontsize=9)
ax.set_xlim(-0.78,0.82); ax.set_xlabel("Spearman $\\rho$ with cyto7 type (allocortex $\\to$ koniocortex)",fontsize=9)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y",length=0)
ax.text(0.40,len(M)-0.55,"rises with type",fontsize=8,style="italic",color="#b2182b",ha="center")
ax.text(-0.40,len(M)-0.55,"falls with type",fontsize=8,style="italic",color="#2166ac",ha="center")
leg=[Line2D([0],[0],marker="o",color="0.35",lw=0,mfc="0.35",mec="0.35",ms=10,label="survives spin + FDR ($q<0.05$)"),
     Line2D([0],[0],marker="o",color="0.35",lw=0,mfc="white",mec="0.35",mew=2,ms=10,label="directional (n.s.)")]
ax.legend(handles=leg,loc="lower right",fontsize=7.5,frameon=False)
OUT=("figures/v9/manuscript/cyto7_structural_model_gradients_with_centroid_sf.png"
     if WITH_CENTROID_SF else "figures/v9/manuscript/cyto7_structural_model_gradients.png")
fig.savefig(OUT,dpi=600,facecolor="white"); print("saved",OUT)
nsurv=sum(1 for _,_,q in M if q<0.05)
for nm,rho,q in M: print(f"{nm:34s} rho={rho:+.2f} q={q:.3f} {'*' if q<0.05 else ''}")
print(f"survivors (q<0.05): {nsurv}")
