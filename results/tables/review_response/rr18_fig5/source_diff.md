# RR18: the full source diff for `scripts/rr3_fig6_bands.py`

Three changes: the footnote string, the dual write so both shipped copies stay byte-identical,
and the `shutil` import the dual write needs.

```diff
diff --git a/scripts/rr3_fig6_bands.py b/scripts/rr3_fig6_bands.py
index d1c950f..b3f9165 100644
--- a/scripts/rr3_fig6_bands.py
+++ b/scripts/rr3_fig6_bands.py
@@ -17,6 +17,8 @@ Nothing is hard-coded, so the panel cannot disagree with the outcome table.
 """
 from __future__ import annotations
 
+import shutil
+
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.colors as mc
@@ -30,6 +32,7 @@ import rr_common as rc
 
 OUTDIR = rc.OUT / "rr3_fig6"
 TABLE = rc.OUT / "rr2_table" / "outcome_table.csv"
+MANUSCRIPT_FIGS = (rc.REPO_ROOT / "manuscript" / "preprint" / "26th_August_2026" / "figures")
 FIGNAME = "cyto7_structural_model_gradients.png"
 DPI = 600
 
@@ -55,8 +58,11 @@ BANDS = [
     ]),
 ]
 
+# No count here on purpose. A total baked into a rendered image cannot be grepped and does
+# not recompile, so regenerating the outcome table silently falsifies the figure. The count
+# lives in the tex caption, which is one grep from the generator that produces it.
 RULE = ("Headline panel, not a census: one measure per construct, from data independent of the atlas.\n"
-        "Individual receptor maps in Fig. S4, disease maps in Fig. S7, all 136 tests in Table S-new.")
+        "Individual receptor maps in Fig. S4, disease maps in Fig. S7, the complete set in Table S5.")
 
 
 def main(argv=None):
@@ -145,6 +151,10 @@ def main(argv=None):
     out = OUTDIR / FIGNAME
     fig.savefig(str(out), dpi=DPI, facecolor="white")
     plt.close(fig)
+    # Write the shipped copy in the same pass. Staging it by hand is how the manuscript copy
+    # and the generator's copy drift apart; the same failure produced the stale table in RR16.
+    shutil.copy2(out, MANUSCRIPT_FIGS / FIGNAME)
+    print(f"staged   {MANUSCRIPT_FIGS / FIGNAME}")
     n_surv = int(df.survives.sum())
     print(f"saved {out} ({len(df)} measures, {n_surv} survivors)")
     for _, r in df.iterrows():
```
