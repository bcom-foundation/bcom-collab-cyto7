# RR19: source diffs

Three files. The first is the task; the second and third are what the cross-reference sweep found.

```diff
diff --git a/scripts/rr2_outcome_table.py b/scripts/rr2_outcome_table.py
index ea65c90..ae126a9 100644
--- a/scripts/rr2_outcome_table.py
+++ b/scripts/rr2_outcome_table.py
@@ -733,7 +733,11 @@ def write_tex(df: pd.DataFrame):
          r"{\footnotesize",
          r"\setlength{\tabcolsep}{4pt}",
          r"\begin{longtable}{" + cw + "}",
-         r"\caption{\textbf{Supplementary Table S5. Complete outcome table.} Every statistical "
+         # S8, not S5: the supplementary tables were renumbered into ascending order of
+         # appearance (they ran S1, S2, S3, S6, S4, S7, S8, S5) and the outcome table is now
+         # last. The shipped manuscript copy already carries S8; this string is what regenerates
+         # it, so it has to match or a re-run silently reverts the caption.
+         r"\caption{\textbf{Supplementary Table S8. Complete outcome table.} Every statistical "
          r"test run in this study, with its hypothesis status, statistical unit, null, FDR family, "
          r"raw $p$, within-family $q$ and effect size. Status records whether a test was "
          r"specified in advance, not whether its result is retained: the connectivity analysis "
@@ -741,7 +745,7 @@ def write_tex(df: pd.DataFrame):
          r"does not survive a topology-preserving null or a bundle-level model. An asterisk "
          r"marks tests surviving at $q<0.05$ within their family. Families are never pooled. "
          r"Tests reported without a $q$ had no FDR applied within their family and are marked "
-         r"accordingly; Table~S6 lists every family with its size, whether correction was "
+         r"accordingly; Table~S4 lists every family with its size, whether correction was "
          r"applied and why.}"
          r"\label{tab:outcomes}\\",
          r"\hline",
diff --git a/scripts/rr3_fig6_bands.py b/scripts/rr3_fig6_bands.py
index b3f9165..d03b671 100644
--- a/scripts/rr3_fig6_bands.py
+++ b/scripts/rr3_fig6_bands.py
@@ -58,11 +58,15 @@ BANDS = [
     ]),
 ]
 
-# No count here on purpose. A total baked into a rendered image cannot be grepped and does
-# not recompile, so regenerating the outcome table silently falsifies the figure. The count
-# lives in the tex caption, which is one grep from the generator that produces it.
+# No count and no table number here, on purpose. Text baked into a rendered image cannot be
+# grepped and does not recompile, so anything that can change underneath it silently falsifies
+# the figure. That already happened twice: the test count went stale when the outcome table grew
+# (RR16), and the table number went stale when the supplement was renumbered (RR19). Both live in
+# the tex caption, which is one grep from the generator that produces them. The table is named,
+# not numbered, because there is only one outcome table and its own caption identifies it.
 RULE = ("Headline panel, not a census: one measure per construct, from data independent of the atlas.\n"
-        "Individual receptor maps in Fig. S4, disease maps in Fig. S7, the complete set in Table S5.")
+        "Individual receptor maps in Fig. S4, disease maps in Fig. S7, the complete outcome table in "
+        "the supplement.")
 
 
 def main(argv=None):
```
