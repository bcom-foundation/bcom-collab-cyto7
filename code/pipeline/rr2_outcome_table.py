"""RR2 - assemble the complete outcome table.

One row per statistical test actually run, harvested from the released CSVs. No
number is recomputed and no released file is modified: this is assembly plus
reconciliation against what the manuscript states. Emits

  outcome_table.csv        machine-readable, every test
  outcome_table.tex        longtable for the supplement, survivors marked *
  survivor_census.csv      per family: members, survivors
  discrepancies.csv        every CSV-versus-manuscript mismatch found

Run::
    conda run -n cyto7 python scripts/rr2_outcome_table.py
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import rr_common as rc
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import cyto7_config as cfg

from cyto7_surface_io import REPO_ROOT

OUTDIR = rc.OUT / "rr2_table"
F = cfg.results_dir("tables")
SF = F / "structure_function"

VERTEX = "vertex (32k fs_LR)"
PARCEL68 = "parcel (Desikan-68)"
SUBJ = "subject (N=89) x vertex (4k fs_LR)"
TYPEPAIR = "type-pair (7x7)"
TYPE7 = "type (7)"

SPIN = "vertex spin (N=1000, seed 0)"
SPIN_PARCEL = "parcel spin (N=1000, seed 0)"
HIER = "hierarchical subject x vertex spin (N=1000, seed 0)"
LABPERM = "label permutation of the 7 type ranks"

rows: list[dict] = []


def add(**kw):
    rows.append(kw)


# --------------------------------------------------------------------------- #
# 1. structure-function family (9 group-average features)
# --------------------------------------------------------------------------- #


def harvest_structure_function():
    t = pd.read_csv(SF / "functional_summary_table_v9.csv")
    one = t.drop_duplicates("FeatureKey")[["FeatureKey", "Feature", "spearman_rho",
                                          "p_spin", "p_spin_fdr"]]
    n_obs = int(t.groupby("FeatureKey")["n"].sum().iloc[0])
    pred = {"myelin": "rise", "thickness": "fall", "gradient": "fall", "timescale": "fall",
            "delta": "fall", "theta": "fall", "alpha": "none", "beta": "rise", "gamma": "rise"}
    for _, r in one.iterrows():
        k = r.FeatureKey
        add(measure=r.Feature, key=k, family="structure_function_9", n_family=len(one),
            status="confirmatory", predicted_direction=f"{pred[k]} (fixed in advance)",
            unit=VERTEX, n_obs=n_obs, null=SPIN,
            effect=f"Spearman rho = {r.spearman_rho:+.4f}",
            p_raw=r.p_spin, q=r.p_spin_fdr, survives="yes" if r.p_spin_fdr < 0.05 else "no",
            source_file="figures/v9/structure_function/functional_summary_table_v9.csv",
            notes=("group-average map; allocortex included in the reported rho "
                   "(allocortex-excluded variant in sensitivity_exclusions.csv)"))


# --------------------------------------------------------------------------- #
# 2. external reference family (3)
# --------------------------------------------------------------------------- #


def harvest_external():
    """External references. RR32: two members, not three.

    BigBrain profile skewness moved out to the BigBrain profile-feature panel, where
    its q was computed all along (RR31 showed its 0.4008 is exactly BH within that
    five-member panel, 0.3207 x 5/4, not BH within this family). With skewness gone
    the remaining two recompute to q = 0.002 and 0.063.
    """
    t = pd.read_csv(SF / "external_validation_table.csv")
    t = t[t.FeatureKey != "bigbrain_profile_skewness"]
    pred = {"genepc1": "rise (fixed in advance)",
            "receptorpc1": "fall (fixed in advance)"}
    for _, r in t.iterrows():
        add(measure=r.Feature, key=r.FeatureKey, family="external_reference_2", n_family=len(t),
            status="confirmatory", predicted_direction=pred.get(r.FeatureKey, "none"),
            unit=VERTEX, n_obs=int(r.n), null=SPIN,
            effect=f"Spearman rho = {r.spearman_rho:+.4f}",
            p_raw=r.p_spin, q=r.p_spin_fdr, survives="yes" if r.p_spin_fdr < 0.05 else "no",
            source_file="figures/v9/structure_function/external_validation_table.csv",
            notes=("gene PC1 from AHBA (abagen, 6 donors, fsaverage 10k -> 32k)"
                   if r.FeatureKey != "receptorpc1" else
                   "PC1 of the 19 Hansen-2022 PET maps"))


def harvest_bigbrain_profiles():
    """RR32 A1: the BigBrain profile-feature panel, declared as its own family.

    Five features over the raw 50-depth intensity profiles. RR31 established that all
    five stored q reproduce from BH within these five, so the family was real all
    along and simply undeclared; the profile-SD test at rho +0.44, q 0.015 was the
    surviving test sitting outside the census.
    """
    p = SF / "bigbrain_profiles" / "profile_features_by_type.csv"
    t = pd.read_csv(p)
    prereg = "bigbrain_profile_skewness"
    for _, r in t.iterrows():
        key = f"bigbrain_profile_{r.FeatureKey}"
        add(measure=f"BigBrain {r.Feature}", key=key,
            family="bigbrain_profile_panel_5", n_family=len(t),
            status="confirmatory" if key == prereg else "exploratory",
            predicted_direction="fall (pre-registered)" if key == prereg else "none",
            unit=VERTEX, n_obs=int(r.n), null=SPIN,
            effect=f"Spearman rho = {r.spearman_rho:+.4f}",
            p_raw=r.p_spin, q=r.p_spin_fdr,
            survives="yes" if r.p_spin_fdr < 0.05 else "no",
            source_file="figures/v9/structure_function/bigbrain_profiles/"
                        "profile_features_by_type.csv",
            notes=("pre-registered differentiation index; its gate failed (|rho| 0.209 "
                   "against a 0.4 target), so the profile leg is reported as inconclusive"
                   if key == prereg else
                   "single-specimen BigBrain profiles; reported, not used as an arbiter"))


# --------------------------------------------------------------------------- #
# 3. MEG dynamics: the operative single-member family and the as-coded families
# --------------------------------------------------------------------------- #


def harvest_meg():
    coded = pd.read_csv(SF / "meg_dynamics_v2_summary.csv").set_index("metric")
    single = pd.read_csv(SF / "meg_dynamics_v2_summary_fdr_singlemember.csv").set_index("metric")
    label = {"int_area": "Intrinsic timescale (area under the ACF, pre-registered)",
             "int_1e": "Intrinsic timescale (1/e lag)",
             "int_tau": "Intrinsic timescale (exponential decay)",
             "knee_tau": "Intrinsic timescale (knee-derived)",
             "exponent_fixed": "Aperiodic exponent (fixed, 2-40 Hz, pre-registered)",
             "exponent_knee": "Aperiodic exponent (knee mode)",
             "exponent_broad": "Aperiodic exponent (broadband 1-100 Hz)",
             "offset": "Spectral offset", "peak_freq": "Peak frequency",
             "centroid": "Power-weighted spectral centroid",
             "sf_ratio": "Slow/fast band-power ratio",
             "osc_delta": "Aperiodic-corrected delta power",
             "osc_theta": "Aperiodic-corrected theta power",
             "osc_alpha": "Aperiodic-corrected alpha power",
             "osc_beta": "Aperiodic-corrected beta power",
             "osc_gamma1": "Aperiodic-corrected gamma power"}
    fam_single = [m for m in single.index if single.loc[m, "role"] == "family_member"]
    robust = [m for m in single.index if single.loc[m, "role"] != "family_member"]
    osc_only = [m for m in coded.index if m not in single.index]
    n_subj = 89
    for m in coded.index:
        c = coded.loc[m]
        if m in fam_single:
            fam, nfam, q = "dynamics_singlemember_7", len(fam_single), float(single.loc[m, "fdr_q_singlemember"])
            status = "confirmatory" if c["primary"] else "confirmatory"
        elif m in robust:
            fam, nfam, q = "dynamics_robustness_variants (outside FDR)", len(robust), np.nan
            status = "robustness"
        else:
            fam, nfam, q = "osc_band_as_coded_4", len(osc_only), float(c["fdr_q"])
            status = "exploratory"
        pd_ = {"neg": "fall", "pos": "rise", "none": "none"}[c["predicted_sign"]]
        add(measure=label.get(m, m), key=m, family=fam, n_family=nfam, status=status,
            predicted_direction=f"{pd_} (fixed in advance)" if c["primary"] else pd_,
            unit=SUBJ, n_obs=f"{n_subj} subjects x 8004 vertices", null=HIER,
            effect=f"mean over {n_subj} subjects of the per-subject vertex-level Spearman rho = "
                   f"{c['group_mean_rho']:+.4f}",
            p_raw=float(c["spin_p"]), q=q,
            survives=("yes" if (np.isfinite(q) and q < 0.05) else
                      ("raw p < 0.05, outside FDR" if c["spin_p"] < 0.05 else "no")),
            source_file=("figures/v9/structure_function/meg_dynamics_v2_summary_fdr_singlemember.csv"
                         if m in single.index else
                         "figures/v9/structure_function/meg_dynamics_v2_summary.csv"),
            notes=(f"as-coded family {c['family']} (q={c['fdr_q']:.4f}); "
                   f"single-member recompute is the operative family per section 2.6"
                   + ("; NEAR-CONSTANT MAP, effect not estimable (see RR5 "
                      "degeneracy_diagnostics.csv)" if m in ("osc_delta", "osc_gamma1") else "")),
            rho_median_subject_meg=float(c["group_median_rho"]),
            frac_subjects_predicted_sign_meg=(float(c["frac_predicted"])
                                              if np.isfinite(c["frac_predicted"]) else np.nan),
            rho_allocortex_excluded_meg=float(c["group_mean_rho_allo_excl"]),
            spin_p_allocortex_excluded_meg=float(c["spin_p_allo_excl"]))


# --------------------------------------------------------------------------- #
# 4. receptors: 19 maps, 8 composites, diversity
# --------------------------------------------------------------------------- #


def harvest_receptors():
    t = pd.read_csv(SF / "receptor_type_association.csv")
    for _, r in t.iterrows():
        add(measure=f"{r.receptor} ({r.system}, {r.klass})", key=f"receptor_{r.receptor}",
            family="receptor_maps_19", n_family=len(t), status="exploratory",
            predicted_direction="none", unit=VERTEX, n_obs=int(r.n), null=SPIN,
            effect=f"Spearman rho = {r.spearman_rho:+.4f}", p_raw=r.spin_p, q=r.fdr_q,
            survives="yes" if r.fdr_q < 0.05 else "no",
            source_file="figures/v9/structure_function/receptor_type_association.csv",
            notes="Hansen-2022 PET collection; allocortex-excluded variant in the same file")
    c = pd.read_csv(SF / "receptor_composites.csv")
    for _, r in c.iterrows():
        add(measure=f"Receptor composite: {r.composite}", key=f"composite_{r.composite}",
            family="receptor_composites_8", n_family=len(c), status="exploratory",
            predicted_direction="rise (Structural-Model prediction)" if r.composite ==
                                "iono_minus_metabo_index" else "none",
            unit=VERTEX, n_obs=58731, null=SPIN,
            effect=f"Spearman rho = {r.spearman_rho:+.4f}", p_raw=r.spin_p, q=r.fdr_q,
            survives="yes" if r.fdr_q < 0.05 else "no",
            source_file="figures/v9/structure_function/receptor_composites.csv",
            notes=f"members: {r.members} ({r.n_members})"
                  + ("; flagged under-powered" if r.underpowered_flag else ""))
    d = pd.read_csv(SF / "receptor_diversity.csv")
    fig9 = pd.read_csv(SF / "fig9_predictions_summary.csv").set_index("metric")
    for _, r in d.iterrows():
        q = (float(fig9.loc["receptor_diversity_H", "fdr_q_fig9"])
             if r.metric == "shannon_entropy_H" else np.nan)
        add(measure=("Receptor diversity (Shannon entropy of 19 maps)"
                     if r.metric == "shannon_entropy_H" else
                     "Receptor diversity (coefficient-of-variation robustness check)"),
            key=f"diversity_{r.metric}",
            family=("fig9_predictions_3" if r.metric == "shannon_entropy_H"
                    else "no FDR family (robustness variant)"),
            n_family=(3 if r.metric == "shannon_entropy_H" else 1),
            status="confirmatory" if r.primary else "robustness",
            predicted_direction="fall (Structural-Model prediction)", unit=VERTEX,
            n_obs=int(r.n), null=SPIN, effect=f"Spearman rho = {r.spearman_rho:+.4f}",
            p_raw=r.spin_p, q=q,
            survives="yes" if (np.isfinite(q) and q < 0.05) else "no",
            source_file="figures/v9/structure_function/receptor_diversity.csv",
            notes="q comes from the three-member Fig.9-prediction family in "
                  "fig9_predictions_summary.csv" if r.primary else "raw spin p only")


# --------------------------------------------------------------------------- #
# 5. layer markers (parcel level)
# --------------------------------------------------------------------------- #


def harvest_layers():
    t = pd.read_csv(F / "definitional" / "layer_marker_genes.csv")
    for _, r in t.iterrows():
        add(measure=f"{r.feature} [{r.kind}, layer {r.layer if isinstance(r.layer, str) else '-'}]",
            key=f"layer_{r.feature}", family="layer_markers_8", n_family=len(t),
            status="confirmatory" if isinstance(r.prereg, str) else "exploratory",
            predicted_direction=("rise (pre-registered)" if isinstance(r.prereg, str)
                                 else "none"),
            unit=PARCEL68, n_obs=int(r.n), null=SPIN_PARCEL,
            effect=f"Spearman rho = {r.spearman_rho:+.4f}", p_raw=r.p_spin, q=r.p_spin_fdr,
            survives="yes" if r.p_spin_fdr < 0.05 else "no",
            source_file="figures/v9/definitional/layer_marker_genes.csv",
            notes="AHBA parcel-level expression; allocortex-excluded variant in the same file "
                  f"(rho {r.spearman_rho_allo_excl:+.3f}, n {int(r.n_allo_excl)})")


# --------------------------------------------------------------------------- #
# 6. Fig.9 predictions (expansion) and disease vulnerability
# --------------------------------------------------------------------------- #


def harvest_predictions_and_disease():
    t = pd.read_csv(SF / "fig9_predictions_summary.csv")
    label = {"receptor_diversity_H": "Receptor diversity (Shannon entropy)",
             "xu2020_evoexp": "Evolutionary cortical expansion (Xu 2020)",
             "hill2010_devexp": "Developmental cortical expansion (Hill 2010)"}
    for _, r in t.iterrows():
        if r.metric == "receptor_diversity_H":
            continue   # already recorded with the receptor rows, same test
        add(measure=label[r.metric], key=r.metric, family="fig9_predictions_3", n_family=len(t),
            status="confirmatory", predicted_direction="fall (Structural-Model prediction)",
            unit=VERTEX, n_obs=58731, null=SPIN,
            effect=f"Spearman rho = {r.spearman_rho:+.4f}", p_raw=r.spin_p, q=r.fdr_q_fig9,
            survives="yes" if r.fdr_q_fig9 < 0.05 else "no",
            source_file="figures/v9/structure_function/fig9_predictions_summary.csv",
            notes=("one hemisphere only, reported as exploratory"
                   if r.metric == "hill2010_devexp" else ""))
    d = pd.read_csv(F / "external" / "disease_vulnerability.csv")
    for _, r in d.iterrows():
        add(measure=f"ENIGMA cortical-thickness effect size: {r.disorder}",
            key=f"disease_{r.disorder}", family="disease_8", n_family=len(d),
            status="exploratory", predicted_direction="fall with type (vulnerability)",
            unit=PARCEL68, n_obs=int(r.n_parcels), null=SPIN_PARCEL,
            effect=f"Spearman rho = {r.spearman_rho:+.4f}", p_raw=r.spin_p, q=r.fdr_q,
            survives="yes" if r.fdr_q < 0.05 else "no",
            source_file="figures/v9/external/disease_vulnerability.csv",
            notes=f"ENIGMA case-control Cohen d, {r.source_key}")
        add(measure=f"ENIGMA {r.disorder}, partialling the functional gradient",
            key=f"disease_partial_{r.disorder}", family="disease_partial_8", n_family=len(d),
            status="robustness", predicted_direction="fall with type (vulnerability)",
            unit=PARCEL68, n_obs=int(r.n_parcels), null=SPIN_PARCEL,
            effect=f"partial Spearman rho = {r.partial_rho_gradient:+.4f}",
            p_raw=r.partial_spin_p, q=r.partial_fdr_q,
            survives="yes" if r.partial_fdr_q < 0.05 else "no",
            source_file="figures/v9/external/disease_vulnerability.csv",
            notes="gradient-specificity control for the same disorder")


# --------------------------------------------------------------------------- #
# 7. tractography
# --------------------------------------------------------------------------- #


def harvest_tractography():
    t = pd.read_csv(F / "tractography" / "tractography_null.csv").iloc[0]
    add(measure="Short-range connectivity vs type-distance (log-linear slope)",
        key="tract_short_slope", family="tractography_5", n_family=5, status="confirmatory",
        predicted_direction="fall (architectonic-type principle, fixed in advance)",
        unit=TYPEPAIR, n_obs=21, null=LABPERM,
        effect=f"log-linear slope = {t.short_slope_obs:+.4f}",
        p_raw=t.slope_perm_p, q=np.nan, survives="yes" if t.slope_perm_p < 0.05 else "no",
        source_file="figures/v9/tractography/tractography_null.csv",
        notes="no FDR applied within this family of 2; RR7 re-examines this result")
    add(measure="Short-range connectivity vs type-distance, contact area partialled out",
        key="tract_partial_contact", family="tractography_5", n_family=5, status="confirmatory",
        predicted_direction="fall (fixed in advance)", unit=TYPEPAIR, n_obs=21,
        null="parametric p on the partial Pearson r",
        effect=f"partial Pearson r = {t.partial_r_typedist_given_contact:+.4f}",
        p_raw=t.partial_p, q=np.nan, survives="yes" if t.partial_p < 0.05 else "no",
        source_file="figures/v9/tractography/tractography_null.csv",
        notes="parametric, not a spatial null; RR7 re-examines this result")
    tr = pd.read_csv(F / "tractography" / "per_type_connectivity_trends.csv")
    for _, r in tr.iterrows():
        add(measure=f"Per-type connectional property: {r.summary}", key=f"tract_{r.summary}",
            family="tractography_per_type_8 (no FDR applied)", n_family=len(tr),
            status="exploratory", predicted_direction="none", unit=TYPE7, n_obs=7,
            null=f"label permutation of the 7 type ranks (N={int(r.n_perm)}, exhaustive)",
            effect=f"Spearman rho = {r.spearman_rho_vs_type:+.4f}", p_raw=r.perm_p, q=np.nan,
            survives="yes (raw p)" if r.perm_p < 0.05 else "no",
            source_file="figures/v9/tractography/per_type_connectivity_trends.csv",
            notes="7 observations; no FDR was applied across these 8 summaries")


# --------------------------------------------------------------------------- #
# 8. benchmark / added value, and the declared negative control
# --------------------------------------------------------------------------- #


def harvest_added_value():
    g = pd.read_csv(F / "added_value_global.csv")
    for _, r in g.iterrows():
        add(measure=f"Added value over the area-level map, global: {r.feature}",
            key=f"av_global_{r.feature}", family="added_value_global_9", n_family=len(g),
            status="confirmatory", predicted_direction="cyto7 higher |rho| (fixed in advance)",
            unit=VERTEX, n_obs=58731, null=SPIN,
            effect=f"delta |rho| = {r.d_rho_abs:+.4f} (cyto7 {r.rho_cyto7:+.3f} vs "
                   f"von-Economo {r.rho_voneconomo:+.3f})",
            p_raw=r.spin_p_cyto_better, q=r.q_fdr, survives="yes" if r.q_fdr < 0.05 else "no",
            source_file="figures/v9/added_value_global.csv",
            notes="one-sided: cyto7 tracks the feature better than the area-level map")
    loc = pd.read_csv(F / "added_value_localized.csv")
    for _, r in loc.iterrows():
        add(measure=f"Win fraction on the disagreement set ({r.subset}): {r.feature}",
            key=f"av_local_{r.feature}_{r.subset}",
            family="added_value_localized_27 (no FDR applied)", n_family=len(loc),
            status="confirmatory", predicted_direction="win fraction > 0.5 (fixed in advance)",
            unit=VERTEX, n_obs=int(r.n), null="binomial test against 0.5",
            effect=f"win fraction = {r.win_frac_cyto7:.4f}", p_raw=r.binom_p, q=np.nan,
            survives="yes (raw p)" if r.binom_p < 0.05 else "no",
            source_file="figures/v9/added_value_localized.csv",
            notes="per-feature win fraction; the aggregate of myelin/thickness/gradient is the "
                  "reported 60.1% and carries the spin null (RR4)")
    for subset, win, p in (("all", 0.6010, 0.001), ("off-by-1", 0.5757, 0.002),
                           ("off-by->=2", 0.7433, 0.001)):
        add(measure=f"Aggregate win fraction on the disagreement set ({subset})",
            key=f"av_aggregate_{subset}", family="added_value_aggregate_3", n_family=3,
            status="confirmatory", predicted_direction="win fraction > chance (fixed in advance)",
            unit=VERTEX, n_obs={"all": 27298, "off-by-1": 23168, "off-by->=2": 4130}[subset],
            null=SPIN + ", win fraction recomputed per rotation",
            effect=f"win fraction = {win:.4f} across myelin/thickness/gradient",
            p_raw=p, q=np.nan, survives="yes",
            source_file="figures/v9/REPORT_v9.md + figures/v9/review_response/rr4_benchmark/",
            notes="reproduced exactly in RR4, which adds the hemisphere holdout and the "
                  "interior-versus-border split")
    bb = pd.read_csv(SF / "bigbrain_addedvalue" / "added_value_global.csv").iloc[0]
    add(measure="Added value over the area-level map with BigBrain histology as arbiter",
        key="av_bigbrain_global", family="bigbrain_added_value_2", n_family=2,
        status="confirmatory", predicted_direction="cyto7 higher |rho| (fixed in advance)",
        unit=VERTEX, n_obs=58731, null=SPIN,
        effect=f"delta |rho| = {bb.d_rho_abs:+.4f} (cyto7 {bb.rho_cyto7:+.3f} vs "
               f"von-Economo {bb.rho_voneconomo:+.3f})",
        p_raw=bb.spin_p_cyto_better, q=bb.q_fdr, survives="no",
        source_file="figures/v9/structure_function/bigbrain_addedvalue/added_value_global.csv",
        notes="single specimen; reported as a null")
    add(measure="Win fraction on the disagreement set with BigBrain histology as arbiter",
        key="av_bigbrain_localized", family="bigbrain_added_value_2", n_family=2,
        status="confirmatory", predicted_direction="win fraction > chance (fixed in advance)",
        unit=VERTEX, n_obs=27298, null=SPIN + ", win fraction recomputed per rotation",
        effect="win fraction = 0.4990 (chance 0.519)", p_raw=0.7403, q=np.nan, survives="no",
        source_file="figures/v9/structure_function/bigbrain_addedvalue/REPORT_bigbrain_added_value.md",
        notes="explicit null: cyto7 does not beat the area-level map against BigBrain histology")
    nc = pd.read_csv(F / "definitional" / "negative_control.csv")
    for _, r in nc.iterrows():
        add(measure=f"Negative control: {r.Feature}", key=f"negctrl_{r.FeatureKey}",
            family="negative_control_2", n_family=len(nc), status="confirmatory",
            predicted_direction="none (expected null, fixed in advance)", unit=VERTEX,
            n_obs=int(r.n), null=SPIN, effect=f"Spearman rho = {r.spearman_rho:+.4f}",
            p_raw=r.p_spin, q=r.p_spin_fdr, survives="yes" if r.p_spin_fdr < 0.05 else "no",
            source_file="figures/v9/definitional/negative_control.csv",
            notes="a surviving result here would indicate the axis is picking up folding geometry")


# --------------------------------------------------------------------------- #
# Discrepancies against the manuscript
# --------------------------------------------------------------------------- #

MANUSCRIPT_CLAIMS = [
    # (location, quantity, manuscript value, csv value, csv source, verdict)
    ("section 3.5, Fig. 4 caption, Fig. 6", "T1w/T2w myelin rho", "+0.58", "+0.5894",
     "functional_summary_table_v9.csv", "rounds to +0.59, not +0.58"),
    ("section 3.5", "cortical thickness q", "0.009", "0.0060",
     "functional_summary_table_v9.csv",
     "manuscript states q ~ 0.009; the released table gives 0.006, which is also what the "
     "abstract's 'all q <~ 0.006' implies"),
    # RR32: this row used to read "consistent", sourced to "REPORT.md analysis 4 /
    # sensitivity_exclusions.csv". sensitivity_exclusions.csv holds neither number - it is
    # the allocortex exclusion, not the agranular one - and no file in the repository holds
    # the agranular pair at all. Asserting agreement against a file that does not contain
    # the values is the same class of error this table exists to catch.
    # RR33 recomputed this on v9 as a first-class test (scripts/rr33_agranular_exclusion.py),
    # so the record side is now a real pair rather than "not stored anywhere". The prose is
    # still wrong, and now wrong in a way that is established rather than suspected.
    ("section 3.5", "myelin rho excluding vs including agranular", "0.62 vs 0.60",
     "0.5851 vs 0.6039 (types 3-7 vs types 2-7, both allocortex-excluded)",
     "figures/v9/review_response/rr33_figures/agranular_exclusion.csv",
     "WRONG IN VALUE AND IN DIRECTION. Recomputed on v9 through the same validity mask and "
     "spin null as every other correlation (N=1000, seed 0; the run reproduces the published "
     "+0.5894 and the released +0.6039 exactly), excluding agranular gives +0.5851 against "
     "+0.6039 including it: the exclusion LOWERS the correlation by 0.019, where the sentence "
     "asserts it raises it by 0.02. The robustness claim survives - both arms clear their spin "
     "null at p = 0.001 and the correlation moves by less than 0.02 - but the printed pair and "
     "its direction do not. Table S3, which the sentence cites, has no with/without-agranular "
     "columns; the citable source is the robustness listing"),
    ("section 3.6, Annex F", "int_area q", "0.014", "0.0140 single-member / 0.0280 as coded",
     "meg_dynamics_v2_summary_fdr_singlemember.csv vs meg_dynamics_v2_summary.csv",
     "both exist; the single-member recompute is the operative family per section 2.6 and the "
     "manuscript uses it"),
    ("section 3.6", "spectral centroid q", "0.014", "0.0140 single-member / 0.0280 as coded",
     "meg_dynamics_v2_summary_fdr_singlemember.csv", "as above"),
    ("section 3.6", "slow/fast ratio q", "0.021", "0.0210 single-member / 0.0280 as coded",
     "meg_dynamics_v2_summary_fdr_singlemember.csv", "as above"),
    ("section 3.6", "aperiodic-corrected beta q", "0.021", "0.0210 single-member / 0.0280 as coded",
     "meg_dynamics_v2_summary_fdr_singlemember.csv", "as above"),
    ("Annex F", "timescale variants reported outside the family", "outside the family",
     "outside the family in the single-member CSV; inside dynamics_core in the as-coded CSV",
     "meg_dynamics_v2_summary_fdr_singlemember.csv",
     "Annex F is correct with respect to the operative single-member family. The RR2 spec's "
     "premise that Annex F is wrong holds only against the superseded as-coded table"),
    ("section 3.6, Annex F, Table S3", "aperiodic-corrected delta and gamma power are treated as "
     "informative nulls", "reported as null results alongside the other band powers",
     "the maps are near-constant: osc_delta is nonzero at 0.2% of subject-vertices and constant "
     "within 56 of 89 subjects, osc_gamma1 nonzero at 3.0% and constant within 4 of 89",
     "figures/v9/review_response/rr5_meg_controls/degeneracy_diagnostics.csv",
     "specparam is fitted over 2-40 Hz, so a delta peak sits at the edge of the fit range and gamma "
     "(30-60 Hz) mostly outside it; these two rows are uninformative rather than reliable nulls and "
     "should be described as not estimable, not as null effects"),
    ("section 3.3 (line 333)", "off-by-2-or-more win fraction", "74.4%", "74.33%",
     "added_value_localized.csv (aggregate of myelin/thickness/gradient)",
     "the released REPORT_v9.md already says 74.3%; 74.4% is a rounding slip repeated twice in "
     "the same sentence"),
    ("section 3.3 (line 333)", "cyto7 vs area-level whole-cortex |rho|, myelin",
     "0.59 vs 0.50", "0.6082 vs 0.4918", "added_value_global.csv",
     "the quoted 0.59 is the nine-feature-panel value from functional_summary_table_v9.csv, which "
     "uses a different mask from the added-value comparison it is quoted inside; the two numbers in "
     "the sentence come from two different vertex sets"),
    ("section 3.3 (line 333)", "cyto7 vs area-level whole-cortex |rho|, thickness",
     "0.50 vs 0.42", "0.5090 vs 0.4130", "added_value_global.csv",
     "same mask mismatch; the released added-value values round to 0.51 vs 0.41"),
    ("section 3.7, Fig. 5 caption", "short-range slope", "-1.23", "-1.2306",
     "tractography_null.csv", "consistent"),
    ("section 3.7", "contact area vs short-range connectivity r", "+0.58", "+0.5754",
     "tractography_null.csv", "consistent"),
    ("section 3.7", "partial r given contact area", "-0.64 (p = 0.002)", "-0.6358 (p = 0.0019)",
     "tractography_null.csv", "consistent"),
    ("section 3.7", "per-type short-range degree per area rho", "-0.89 (p = 0.01)",
     "-0.8929 (p = 0.0123)", "per_type_connectivity_trends.csv", "consistent"),
    ("section 3.8", "four surviving receptor maps, all q", "0.043", "0.0427",
     "receptor_type_association.csv", "consistent"),
    ("section 3.8", "ionotropic-to-metabotropic index", "+0.43 (q = 0.016)", "+0.4263 (q = 0.0160)",
     "receptor_composites.csv", "consistent"),
    ("section 3.8, Fig. 6 caption", "number of surviving measures in Fig. 6", "six",
     "26 tests have q < 0.05 across all families (31 including the disease family)",
     "this table", "true only under an unstated de-duplication rule; RR3 states the rule in "
                   "the figure and reports ten"),
    # q restated after RR13: the published q of 0.000 came from ENIGMA's convention,
    # which section 2.6 does not describe; under the stated convention it is 0.005
    ("Fig. 6 caption", "RORB omitted from Fig. 6", "not shown",
     "+0.7761 (q = 0.005 under the section 2.6 convention, RR13), the strongest "
     "measure in the paper",
     "layer_marker_genes.csv + rr13_pvalues/layer_marker_corrected.csv", "RR3 adds it"),
]


# --------------------------------------------------------------------------- #
# Convention corrections (RR13, RR13b)
# --------------------------------------------------------------------------- #

#: RR13 established that two families computed their spin p with ENIGMA's
#: perm_sphere_p, which omits the plus-one correction and is one-tailed, so the
#: values they published are not obtainable under the convention section 2.6
#: states. RR13 and RR13b recomputed both families on the same rotations and
#: seed. The published CSVs are left untouched, as the guardrails require, so the
#: corrected values are read from the review-response outputs here instead.
RR13_LAYERS = rc.OUT / "rr13_pvalues" / "layer_marker_corrected.csv"
RR13B_DISEASE = rc.OUT / "rr13b_disease" / "disease_convention_before_after.csv"


def harvest_missing_tractography():
    """Two Annex G results that the table claimed to list but did not.

    Both are read from the RR7 and RR9 outputs rather than retyped.
    """
    import json as _json
    rr7 = _json.loads((rc.OUT / "rr7_tracto" / "rr7_summary.json").read_text(encoding="utf-8"))
    nul = rr7["published_slope_vs_topology_preserving_null"]
    add(measure="Short-range connectivity vs type-distance, topology-preserving null",
        key="tract_rotation_null", family="tractography_5", n_family=5,
        status="confirmatory", predicted_direction="fall with type-distance",
        unit="type pair (7x7 off-diagonal), n=21",
        n_obs=21, null="type-map rotation (N=1000, seed 0), scale-free statistic",
        effect=f"Pearson rho = {nul['observed_pearson_r']:+.4f} "
               f"(null mean {nul['spin_null_mean_pearson_r']:+.4f})",
        p_raw=nul["p_spin_pearson_r"], q=np.nan, survives="no",
        source_file="figures/v9/review_response/rr7_tracto/rr7_summary.json",
        notes="the published label-permutation p of 0.003 becomes 0.040 under a null that "
              "preserves the concentric arrangement; the null is centred at a third of the "
              "observed effect")

    mc = pd.read_csv(rc.OUT / "rr7_tracto" / "model_coefficients.csv")
    s1 = mc[(mc["design"].str.startswith("bundle x type-pair, zero-completed")) &
            (mc["spec"].str.startswith("S1"))].iloc[0]
    add(measure="Short-range connectivity vs type-distance, bundle level, endpoint-pair offset",
        key="tract_bundle_endpoint_offset", family="tractography_5", n_family=5,
        status="confirmatory", predicted_direction="fall with type-distance",
        unit="bundle x type pair, n=924",
        n_obs=int(s1["n"]), null="negative binomial, tract-clustered errors",
        effect=f"beta = {float(s1['coef']):+.4f} per ordinal step",
        p_raw=float(s1["p_model"]), q=np.nan, survives="no",
        source_file="figures/v9/review_response/rr7_tracto/model_coefficients.csv",
        notes="the 21 off-diagonal cells summarise 33 template bundles; at the level the data "
              "exist the association is absent")

    r9 = pd.read_csv(rc.OUT / "rr9_proximity" / "model_coefficients_rr9.csv")
    b1 = r9[(r9["level"] == "bundle") & (r9["model"] == "R1")].iloc[0]
    add(measure="Short-range connectivity vs type-distance, bundle level, 80 mm opportunity offset",
        key="tract_bundle_opportunity_offset", family="tractography_5", n_family=5,
        status="confirmatory", predicted_direction="fall with type-distance",
        unit="bundle x type pair, n=924",
        n_obs=int(b1["n"]), null="negative binomial, tract-clustered errors",
        effect=f"beta = {float(b1['coef']):+.4f} per ordinal step",
        p_raw=float(b1["p_model"]), q=np.nan, survives="no",
        source_file="figures/v9/review_response/rr9_proximity/model_coefficients_rr9.csv",
        notes="offsetting by the vertex pairs actually within 80 mm rather than by all possible "
              "pairs leaves the bundle-level coefficient at zero")


def harvest_benchmark_controls():
    """RR14's comparator and granularity-matched-null controls on the section 3.3 benchmark."""
    import json as _json
    d = _json.loads((rc.OUT / "rr14_comparator" / "rr14_summary.json").read_text(encoding="utf-8"))
    mn = d["matched_null"]
    add(measure="Win fraction vs the granularity-matched null (200 size-matched partitions)",
        key="bench_matched_null", family="benchmark_controls_4", n_family=4,
        status="robustness", predicted_direction="exceed the matched baseline",
        unit=VERTEX, n_obs=int(d["n_disagreement"]),
        null=f"{mn['n']} contiguous size-matched seven-class partitions",
        effect=f"win fraction = {mn['cyto7_observed']:.4f} (matched-null mean "
               f"{mn['mean']:.4f}, s.d. {mn['sd']:.4f}, {mn['cyto7_percentile']:.1f}th percentile)",
        p_raw=mn["p_vs_matched_null"], q=np.nan, survives="yes",
        source_file="figures/v9/review_response/rr14_comparator/rr14_summary.json",
        notes="replaces the rotated-cyto7 chance level of 0.46, which the matched null puts at "
              "its 0.5th percentile")
    for w in d["win_fractions"]:
        ch = w["challenger"]
        if ch.startswith("cyto7"):
            continue
        add(measure=f"Win fraction of the {ch} as challenger against the area-level map",
            key="bench_" + ch.replace(" ", "_"), family="benchmark_controls_4", n_family=4,
            status="robustness", predicted_direction="none (comparator)",
            unit=VERTEX, n_obs=int(w["n_disagreement_vertices"]),
            null="none (descriptive comparator on the same disagreement set)",
            effect=f"win fraction = {w['win_fraction']:.4f}",
            p_raw=np.nan, q=np.nan, survives="n/a",
            source_file="figures/v9/review_response/rr14_comparator/winfraction_comparators.csv",
            notes="scored on the same 27,298 vertices as cyto7; each septile is scored largely "
                  "against its own defining arbiter")


def apply_convention_corrections(df: pd.DataFrame, log=print) -> tuple:
    """Replace p and q for the two families that changed convention.

    Only p_raw, q and survives move. The effect sizes, units, families and
    statuses are unchanged, because only the null convention changed.
    """
    changes = []

    if RR13_LAYERS.exists():
        lay = pd.read_csv(RR13_LAYERS)
        # RR13's primary reading: plus-one, two-tailed, at the stated N = 1000,
        # averaged over the two null directions. This is the reading whose values
        # the manuscript already quotes (RORB p = 0.003, q = 0.005).
        for _, r in lay.iterrows():
            key = f"layer_{r['feature']}"
            m = df["key"] == key
            if not m.any():
                log(f"    [WARN] no row for {key}")
                continue
            old_p = float(df.loc[m, "p_raw"].iloc[0])
            old_q = float(df.loc[m, "q"].iloc[0])
            new_p = float(r["p_stated_2tail_plus1_N1000_avg"])
            new_q = float(r["q_stated_2tail_plus1_N1000_avg"])
            df.loc[m, "p_raw"] = new_p
            df.loc[m, "q"] = new_q
            df.loc[m, "survives"] = "yes" if new_q < 0.05 else "no"
            df.loc[m, "notes"] = df.loc[m, "notes"].astype(str) + (
                "; p and q recomputed under the section 2.6 convention (RR13), "
                "the published file used ENIGMA perm_sphere_p")
            changes.append({"key": key, "family": "layer_markers_8",
                            "p_old": old_p, "p_new": new_p,
                            "q_old": old_q, "q_new": new_q,
                            "survives_old": "yes" if old_q < 0.05 else "no",
                            "survives_new": "yes" if new_q < 0.05 else "no"})
    else:
        log(f"    [WARN] {RR13_LAYERS} missing; layer-marker rows left uncorrected")

    if RR13B_DISEASE.exists():
        dis = pd.read_csv(RR13B_DISEASE)
        for _, r in dis.iterrows():
            key = f"disease_{r['disorder']}"
            m = df["key"] == key
            if not m.any():
                log(f"    [WARN] no row for {key}")
                continue
            old_p = float(df.loc[m, "p_raw"].iloc[0])
            old_q = float(df.loc[m, "q"].iloc[0])
            new_p = float(r["spin_p_stated_convention"])
            new_q = float(r["fdr_q_stated_convention"])
            df.loc[m, "p_raw"] = new_p
            df.loc[m, "q"] = new_q
            df.loc[m, "survives"] = "yes" if new_q < 0.05 else "no"
            df.loc[m, "notes"] = df.loc[m, "notes"].astype(str) + (
                "; p and q recomputed under the section 2.6 convention (RR13b), "
                "the published file used ENIGMA perm_sphere_p")
            changes.append({"key": key, "family": "disease_8",
                            "p_old": old_p, "p_new": new_p,
                            "q_old": old_q, "q_new": new_q,
                            "survives_old": "yes" if old_q < 0.05 else "no",
                            "survives_new": "yes" if new_q < 0.05 else "no"})
    else:
        log(f"    [WARN] {RR13B_DISEASE} missing; disease rows left uncorrected")

    ch = pd.DataFrame(changes)
    log(f"  applied {len(ch)} convention corrections "
        f"({int((ch['survives_old'] != ch['survives_new']).sum()) if len(ch) else 0} "
        f"change survival status)")
    return df, ch


# --------------------------------------------------------------------------- #
# RR32 A5: the defect this generator had, and the guard against it recurring
# --------------------------------------------------------------------------- #

# Families that declare no FDR (Table S4). A q is not expected for these.
NO_FDR_FAMILIES = {
    "added_value_aggregate_3", "added_value_localized_27 (no FDR applied)",
    "tractography_5", "tractography_per_type_8 (no FDR applied)",
    "dynamics_robustness_variants (outside FDR)", "benchmark_controls_4",
    "no FDR family (robustness variant)",
}

# Rows whose upstream q is knowingly drawn from a different family than the one this
# table declares. Each needs a reason, and the recomputed value is what gets printed.
# Anything NOT registered here that disagrees aborts the build: a generator that can
# print a q inconsistent with its own family label is how RR31's errors happened.
EXPECTED_UPSTREAM_MISMATCH = {
    "osc_delta": "upstream q is BH within the seven-member as-coded dynamics family; "
                 "RR32 A3 makes these four a family in their own right",
    "osc_theta": "as osc_delta",
    "osc_alpha": "as osc_delta",
    "osc_beta": "as osc_delta",
    "osc_gamma1": "as osc_delta",
    "av_bigbrain_global": "upstream added_value_global.csv stores q equal to its own raw p, "
                          "i.e. uncorrected; RR32 A4 makes this a two-member family",
    "av_bigbrain_localized": "upstream stores no q; RR32 A4 makes this a two-member family",
    "genepc1": "upstream q is BH within the old three-member external family; RR32 A2 "
               "drops skewness, so the remaining two recompute",
    "receptorpc1": "as genepc1",
}


def _bh_q(pvals):
    """Benjamini-Hochberg step-up over a family, returned in input order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def reconcile_family_fdr(df):
    """Recompute BH inside each declared family and fail loudly on a surprise.

    The defect RR31 found: this generator read q from whichever upstream CSV produced
    a row, and never checked it against the family it labelled. So a row could carry a
    q computed over seven tests while the table said four, and nothing noticed.

    Now the declared family plus the stored p is the authority. Every corrected family
    is recomputed; every disagreement with the upstream value must be registered above
    with a reason, or the build stops.
    """
    df = df.copy()
    report = []
    surprises = []
    for fam, sub in df.groupby("family", sort=False):
        n_listed = len(sub)
        declared = sorted({int(v) for v in sub["n_family"].dropna().unique()})
        if len(declared) > 1:
            surprises.append(f"family {fam!r} labels its size inconsistently across rows: "
                             f"{declared}")
        for d in declared:
            if d != n_listed:
                surprises.append(f"family {fam!r} labels n={d} but lists {n_listed} rows")
        if fam in NO_FDR_FAMILIES:
            continue
        ps = pd.to_numeric(sub["p_raw"], errors="coerce")
        if ps.isna().any():
            surprises.append(
                f"family {fam!r} declares FDR but {int(ps.isna().sum())} of {n_listed} "
                f"rows have no raw p, so q cannot be computed")
            continue
        newq = _bh_q(ps.values)
        for (idx, row), q_new in zip(sub.iterrows(), newq):
            q_old = pd.to_numeric(pd.Series([row["q"]]), errors="coerce").iloc[0]
            changed = pd.isna(q_old) or abs(float(q_old) - float(q_new)) > 5e-4
            if changed:
                key = row["key"]
                if key not in EXPECTED_UPSTREAM_MISMATCH:
                    surprises.append(
                        f"{fam}/{key}: upstream q={q_old} disagrees with BH within the "
                        f"declared family ({q_new:.4f}), and is not a registered exception")
                report.append({"family": fam, "key": key, "p": float(ps.loc[idx]),
                               "q_upstream": None if pd.isna(q_old) else float(q_old),
                               "q_recomputed": float(q_new),
                               "reason": EXPECTED_UPSTREAM_MISMATCH.get(key, "UNREGISTERED")})
            df.at[idx, "q"] = float(q_new)
            df.at[idx, "survives"] = "yes" if float(q_new) < 0.05 else "no"

    if surprises:
        print("\n  FDR RECONCILIATION FAILED:")
        for s in surprises:
            print(f"    {s}")
        raise SystemExit(
            "Family FDR reconciliation failed: a q disagrees with the family it is "
            "labelled with, and the disagreement is not registered. Fix the family "
            "assignment or register the exception with a reason; do not silently "
            "prefer either value.")
    if report:
        print(f"  FDR reconciled: {len(report)} q recomputed within the declared family")
        for r in report:
            print(f"    {r['family']}/{r['key']}: {r['q_upstream']} -> {r['q_recomputed']:.4f}")
    return df, pd.DataFrame(report)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    frozen_before = rc.frozen_hashes()

    harvest_structure_function()
    harvest_external()
    harvest_bigbrain_profiles()
    harvest_meg()
    harvest_receptors()
    harvest_layers()
    harvest_predictions_and_disease()
    harvest_tractography()
    harvest_added_value()
    harvest_missing_tractography()
    harvest_benchmark_controls()

    cols = ["measure", "key", "family", "n_family", "status", "predicted_direction", "unit",
            "n_obs", "null", "effect", "effect_value", "p_raw", "q", "survives",
            "source_file", "notes",
            "rho_median_subject_meg", "frac_subjects_predicted_sign_meg",
            "rho_allocortex_excluded_meg", "spin_p_allocortex_excluded_meg"]
    df = pd.DataFrame(rows)
    # numeric effect, so downstream figures read the value instead of re-deriving it
    df["effect_value"] = df["effect"].str.extract(r"=\s*([+-]?\d*\.?\d+)").astype(float)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    df, corrections = apply_convention_corrections(df)
    df, reconciliation = reconcile_family_fdr(df)
    if len(corrections):
        corrections.to_csv(OUTDIR / "convention_corrections.csv", index=False)
    df.to_csv(OUTDIR / "outcome_table.csv", index=False)
    print(f"  {len(df)} tests, {df.family.nunique()} families")

    # ---------------- survivor census ---------------- #
    cen = []
    for fam, sub in df.groupby("family", sort=False):
        surv = int((sub["survives"].astype(str).str.startswith("yes")).sum())
        strict = int(((sub["q"] < 0.05) & sub["q"].notna()).sum())
        cen.append({"family": fam, "n_members": int(sub["n_family"].iloc[0]),
                    "n_tests_listed": len(sub), "n_survivors_q_lt_0.05": strict,
                    "n_survivors_incl_raw_p": surv,
                    "fdr_applied": "yes" if sub["q"].notna().any() else "no"})
    census = pd.DataFrame(cen)
    census.loc[len(census)] = {"family": "TOTAL", "n_members": census["n_members"].sum(),
                               "n_tests_listed": int(census["n_tests_listed"].sum()),
                               "n_survivors_q_lt_0.05": int(census["n_survivors_q_lt_0.05"].sum()),
                               "n_survivors_incl_raw_p": int(census["n_survivors_incl_raw_p"].sum()),
                               "fdr_applied": ""}
    census.to_csv(OUTDIR / "survivor_census.csv", index=False)
    print(census.to_string(index=False))

    disc = pd.DataFrame(MANUSCRIPT_CLAIMS, columns=[
        "manuscript_location", "quantity", "manuscript_value", "csv_value", "csv_source",
        "verdict"])
    disc["is_discrepancy"] = ~disc["verdict"].str.startswith("consistent")
    disc.to_csv(OUTDIR / "discrepancies.csv", index=False)
    print(f"\n  {int(disc.is_discrepancy.sum())} discrepancies of "
          f"{len(disc)} manuscript values checked")

    write_tex(df)

    ok, _detail = rc.check_frozen(frozen_before)
    (OUTDIR / "rr2_summary.json").write_text(json.dumps({
        "n_tests": len(df), "n_families": int(df.family.nunique()),
        "n_survivors_q_lt_0.05": int(((df["q"] < 0.05) & df["q"].notna()).sum()),
        "n_survivors_excl_disease": int(((df["q"] < 0.05) & df["q"].notna() &
                                         ~df["family"].str.startswith("disease")).sum()),
        "n_convention_corrections": int(len(corrections)),
        "frozen_unchanged": ok}, indent=2), encoding="utf-8")
    print("frozen files unchanged:", ok)
    return df, census, disc


def _esc(s) -> str:
    s = "" if s is None else str(s)
    # the shipped table breaks long family names, so an escaped underscore carries
    # an \allowbreak with it
    for a, b in (("\\", "\\textbackslash{}"), ("_", "\\_\\allowbreak{}"), ("&", "\\&"),
                 ("%", "\\%"),
                 ("#", "\\#"), ("$", "\\$"), (">=", "$\\geq$"), ("<", "$<$"), (">", "$>$")):
        s = s.replace(a, b)
    return s


def _floor_from_null(null_str) -> float | None:
    """Smallest p the family's null can produce, read from its own null description.

    Section 2.6's plus-one convention puts the floor at 1/(N+1), so a family that
    states its N states its floor. Families with no N are parametric or binomial
    and have no resampling floor, so their p may legitimately underflow.
    """
    m = re.search(r"N\s*=\s*([0-9,]+)", str(null_str))
    if not m:
        return None
    return 1.0 / (int(m.group(1).replace(",", "")) + 1)


def _fmt_pq(v, floor) -> str:
    """Never emit 0.000: it is impossible under the plus-one convention.

    Values that would round to zero at three decimals are shown as $<$0.001,
    except in a family whose own floor is finer than that, where the extra digit
    is real and is printed instead of being thrown away.
    """
    if v is None or not np.isfinite(v):
        return "--"
    v = float(v)
    if v >= 0.0005:
        return f"{v:.3f}"
    if floor is not None and floor < 0.0005:
        s = f"{v:.4f}"
        if float(s) > 0:
            return s
    return r"$<$0.001"


def write_tex(df: pd.DataFrame):
    # column spec, wrapper and caption match the file the supplement includes; the
    # supplement wraps this in landscape, so no page geometry is set here
    cw = "".join(r">{\raggedright\arraybackslash}p{%s}" % w for w in
                 ("6.0cm", "3.0cm", "1.5cm", "3.8cm", "4.0cm", "1.2cm", "1.0cm", "0.5cm"))
    L = [r"% Complete outcome table (RR2). Standalone: no custom macros.",
         r"{\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{longtable}{" + cw + "}",
         # S8, not S5: the supplementary tables were renumbered into ascending order of
         # appearance (they ran S1, S2, S3, S6, S4, S7, S8, S5) and the outcome table is now
         # last. The shipped manuscript copy already carries S8; this string is what regenerates
         # it, so it has to match or a re-run silently reverts the caption.
         r"\caption{\textbf{Supplementary Table S8. Outcome table: the declared FDR families.} "
         r"Every test in the declared FDR families, with its hypothesis status, statistical unit, null, FDR family, "
         r"raw $p$, within-family $q$ and effect size. Status records whether a test was "
         r"specified in advance, not whether its result is retained: the connectivity analysis "
         r"of Annex~G was specified in advance but is reported as exploratory there, because it "
         r"does not survive a spin null or a bundle-level model. An asterisk "
         r"marks tests surviving at $q<0.05$ within their family. Families are never pooled. "
         r"Tests reported without a $q$ had no FDR applied within their family and are marked "
         r"accordingly; Table~S4 lists every family with its size, whether correction was "
         r"applied and why. Robustness and sensitivity analyses reported in the text are not "
         r"confirmatory tests, sit in no declared family and are therefore not listed here; "
         r"they are disclosed, without FDR, in the robustness listing.}"
         r"\label{tab:outcomes}\\",
         r"\hline",
         r"Measure & Family ($n$) & Status & Unit ($n_{\mathrm{obs}}$) & Null & $p_{\mathrm{raw}}$ & "
         r"$q$ & \\",
         r"\hline",
         r"\endfirsthead",
         r"\hline",
         r"Measure & Family ($n$) & Status & Unit ($n_{\mathrm{obs}}$) & Null & $p_{\mathrm{raw}}$ & "
         r"$q$ & \\",
         r"\hline",
         r"\endhead",
         r"\hline",
         r"\endfoot"]
    for _, r in df.iterrows():
        fl = _floor_from_null(r["null"])
        q = _fmt_pq(pd.to_numeric(r["q"], errors="coerce"), fl)
        p = _fmt_pq(pd.to_numeric(r["p_raw"], errors="coerce"), fl)
        star = "*" if str(r["survives"]).startswith("yes") else ""
        eff = _esc(r["effect"]).replace("rho", r"$\rho$")
        L.append(" & ".join([
            _esc(r["measure"]) + r"\newline\footnotesize " + eff,
            _esc(f"{r['family']} ({r['n_family']})"),
            _esc(r["status"]),
            _esc(f"{r['unit']}, n={r['n_obs']}"),
            _esc(r["null"]), p, q, star]) + r" \\")
    L += [r"\hline", r"\end{longtable}", r"}"]
    # explicit LF: the supplement's copy is LF, and letting Windows write CRLF here
    # turns a 16-row change into a whole-file diff once the file is copied across
    text = "\n".join(L)
    (OUTDIR / "outcome_table.tex").write_text(text, encoding="utf-8", newline="\n")
    print(f"  wrote {OUTDIR / 'outcome_table.tex'}")
    # outcome_table.tex is generated output and is the one .tex this pipeline owns;
    # writing both copies here is what stops the manuscript copy drifting from the
    # generator, which is how 25 hand-fixed cells were lost once already
    staged = (REPO_ROOT / "manuscript" / "preprint" / "26th_August_2026"
              / "outcome_table.tex")
    if staged.parent.is_dir():
        staged.write_text(text, encoding="utf-8", newline="\n")
        print(f"  wrote {staged}")


if __name__ == "__main__":
    main()
