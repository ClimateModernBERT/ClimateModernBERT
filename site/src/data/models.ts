import type { CorpusKey, Stage } from "./naming";
import { corpusPhrase, paperNotation, stageShort } from "./naming";

/**
 * CANONICAL MODEL MANIFEST — the single source of truth for the model explorer,
 * docs/model-inventory.md and huggingface/manifests/models.json.
 *
 * Nothing in the UI hard-codes a checkpoint. Add or correct a model here.
 *
 * Evidence policy
 * ---------------
 * `evidence` records HOW a mapping was established. Three levels are used:
 *   "collection"  — listed in the curated "ClimateModernBERT" HF collection
 *                   ("Final Models from the paper").
 *   "mergekit"    — read directly from the repo's mergekit_config.yml, so the
 *                   component checkpoints are known exactly.
 *   "name"        — inferred from the historical repo name only.
 * `ambiguous: true` means two sources of evidence disagree, or the paper does
 * not pin the checkpoint down. Those are surfaced in the UI and must be
 * confirmed by an author — they are NOT silently resolved.
 */

export type Status =
  | "recommended"   // headline model for general use
  | "paper"         // a configuration analyzed in the manuscript
  | "experimental"  // released, but no reported result in the manuscript
  | "legacy"        // earlier pipeline stage, kept for provenance
  | "superseded";   // an equivalent, more complete checkpoint exists

export type Family = "joint" | "merged";

export interface ModelEntry {
  hf_id: string;
  display_name: string;
  /** Paper corpus set. Empty for merges whose effective coverage is noted instead. */
  canonical_corpora: CorpusKey[];
  stage: Stage;
  legacy_stage: "CX" | "CX_LRD";
  family: Family;
  status: Status;
  /** Paper symbol, e.g. "{A, S}" or "θSoup". Null when the paper reports no such model. */
  paper_variant: string | null;
  /** Average F1 from the manuscript, or null when the manuscript reports none. */
  avg_f1: number | null;
  merge_method?: string;
  /** Component repo ids, read from mergekit_config.yml. */
  merge_components?: string[];
  synthetic_generator?: string | null;
  superseded_by?: string | null;
  duplicate_of?: string | null;
  evidence: ("collection" | "mergekit" | "name")[];
  ambiguous?: boolean;
  notes: string;
}

const A: CorpusKey[] = ["A"];
const S: CorpusKey[] = ["S"];
const F: CorpusKey[] = ["F"];
const AS: CorpusKey[] = ["A", "S"];
const AF: CorpusKey[] = ["A", "F"];
const SF: CorpusKey[] = ["S", "F"];
const ASF: CorpusKey[] = ["A", "S", "F"];

/** The three single-source Phase-2 checkpoints that every released merge is built from. */
const SOUP_COMPONENTS = [
  "sraj/CMB_MARK_CX_LRD",
  "sraj/CMB_FWEdu_V2_FastTxt_CX_LRD",
  "sraj/CMB_WX_SYN_CX_LRD",
];
const LEGACY_MERGE_COMPONENTS = [
  "sraj/CMB_MARK_CX_LRD",
  "sraj/CMB_FWEdu_V2_CX_LRD",
  "sraj/CMB_WX_SYN_CX_LRD",
];

export const models: ModelEntry[] = [
  // ───────────────────────── Merged models ─────────────────────────
  {
    hf_id: "sraj/Merge_Linear",
    display_name: "ClimateModernBERT · Soup",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "recommended", paper_variant: "θSoup", avg_f1: 76.3,
    merge_method: "Linear (uniform weight averaging, normalized)",
    merge_components: SOUP_COMPONENTS,
    evidence: ["collection", "mergekit"],
    notes:
      "The manuscript's best model (76.3 average F1). Uniform average of the three single-source Phase-2 checkpoints; the component list was read from the repo's mergekit_config.yml and matches the paper's ℱ (FastText-filtered) component. Start here.",
  },
  {
    hf_id: "sraj/Merge_Linear_NormBalanced_CX_only",
    display_name: "ClimateModernBERT · Norm-balanced Soup (Phase-1 components)",
    canonical_corpora: ASF, stage: "phase1", legacy_stage: "CX",
    family: "merged", status: "paper", paper_variant: "θNorm (CX)", avg_f1: 75.9,
    merge_method: "Linear, weights inverse to task-vector L2 norm (𝒜 2.7 · ℱ 1.0 · 𝒮 2.7)",
    merge_components: ["sraj/CMB_MARK_CX", "sraj/CMB_FWEdu_V2_FastTxt_CX", "sraj/CMB_WX_SYN_CX"],
    evidence: ["collection", "mergekit"],
    notes:
      "Appendix F, Table 10. The strongest Phase-1 merge — merging does not depend on LRD specialization to work.",
  },
  {
    hf_id: "sraj/TA_Lambda10_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD",
    display_name: "ClimateModernBERT · Task Arithmetic (λ = 1.0)",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "paper", paper_variant: "θTA(1.0)", avg_f1: 75.7,
    merge_method: "Task Arithmetic, λ = 1.0",
    merge_components: LEGACY_MERGE_COMPONENTS,
    evidence: ["mergekit", "name"], ambiguous: true,
    notes:
      "Repo name and mergekit config record CMB_FWEdu_V2_CX_LRD as the ℱ component — the pre-FastText web checkpoint — whereas θSoup uses the FastText-filtered one. Correspondence to Table 4 is by merge method and λ, not by verified component set.",
  },
  {
    hf_id: "sraj/TIES_D07_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD",
    display_name: "ClimateModernBERT · TIES (d = 0.7)",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "paper", paper_variant: "θTIES(0.7)", avg_f1: 75.6,
    merge_method: "TIES-Merging, drop ratio 0.7",
    merge_components: LEGACY_MERGE_COMPONENTS,
    evidence: ["mergekit", "name"], ambiguous: true,
    notes: "Same ℱ-component caveat as θTA above.",
  },
  {
    hf_id: "sraj/TIES_D05_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD",
    display_name: "ClimateModernBERT · TIES (d = 0.5)",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "paper", paper_variant: "θTIES(0.5)", avg_f1: 75.4,
    merge_method: "TIES-Merging, drop ratio 0.5",
    merge_components: LEGACY_MERGE_COMPONENTS,
    evidence: ["mergekit", "name"], ambiguous: true,
    notes: "Same ℱ-component caveat as θTA above.",
  },
  {
    hf_id: "sraj/DARE_TIES_D05_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD",
    display_name: "ClimateModernBERT · DARE-TIES (d = 0.5)",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "paper", paper_variant: "θDARE(0.5)", avg_f1: 74.7,
    merge_method: "DARE-TIES, drop ratio 0.5",
    merge_components: LEGACY_MERGE_COMPONENTS,
    evidence: ["mergekit", "name"], ambiguous: true,
    notes: "Same ℱ-component caveat as θTA above.",
  },
  {
    hf_id: "sraj/Merge_Linear_NormBalanced",
    display_name: "ClimateModernBERT · Norm-balanced Soup",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "paper", paper_variant: "θNorm", avg_f1: 74.6,
    merge_method: "Linear, weights inverse to task-vector L2 norm (𝒜 2.7 · ℱ 1.0 · 𝒮 2.7)",
    merge_components: SOUP_COMPONENTS,
    evidence: ["collection", "mergekit"],
    notes:
      "Appendix F, Table 10. Down-weights ℱ, whose task vector has the largest norm (≈43 per layer vs ≈16 for 𝒜 and 𝒮). Does not beat the uniform Soup.",
  },
  {
    hf_id: "sraj/DARE_TIES_D07_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD",
    display_name: "ClimateModernBERT · DARE-TIES (d = 0.7)",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "paper", paper_variant: "θDARE(0.7)", avg_f1: 74.3,
    merge_method: "DARE-TIES, drop ratio 0.7",
    merge_components: LEGACY_MERGE_COMPONENTS,
    evidence: ["mergekit", "name"], ambiguous: true,
    notes: "Same ℱ-component caveat as θTA above.",
  },
  {
    hf_id: "sraj/Merge_Linear_CX_only",
    display_name: "ClimateModernBERT · Soup (Phase-1 components)",
    canonical_corpora: ASF, stage: "phase1", legacy_stage: "CX",
    family: "merged", status: "paper", paper_variant: "θSoup (CX)", avg_f1: 74.2,
    merge_method: "Linear (uniform weight averaging, normalized)",
    merge_components: ["sraj/CMB_MARK_CX", "sraj/CMB_FWEdu_V2_FastTxt_CX", "sraj/CMB_WX_SYN_CX"],
    evidence: ["collection", "mergekit"],
    notes: "Appendix F, Table 10 — the LRD-free counterpart of θSoup.",
  },
  {
    hf_id: "sraj/TA_Lambda05_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD",
    display_name: "ClimateModernBERT · Task Arithmetic (λ = 0.5)",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "paper", paper_variant: "θTA(0.5)", avg_f1: 73.6,
    merge_method: "Task Arithmetic, λ = 0.5",
    merge_components: LEGACY_MERGE_COMPONENTS,
    evidence: ["mergekit", "name"], ambiguous: true,
    notes: "Weakest merge in Table 4. Same ℱ-component caveat as θTA(1.0).",
  },

  // Drop-one Soup ablations (Figure 2)
  {
    hf_id: "sraj/Merge_Drop_MARK_FastText",
    display_name: "Drop-one Soup · without Academic",
    canonical_corpora: SF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "paper", paper_variant: "Soup({𝒮, ℱ})", avg_f1: null,
    merge_method: "Linear (uniform), 𝒜 removed",
    merge_components: ["sraj/CMB_FWEdu_V2_FastTxt_CX_LRD", "sraj/CMB_WX_SYN_CX_LRD"],
    evidence: ["mergekit"],
    notes:
      "Figure 2. Removing 𝒜 costs 4.0 average F1 relative to the full Soup — by far the largest drop-one degradation.",
  },
  {
    hf_id: "sraj/Merge_Drop_SYN_FastText",
    display_name: "Drop-one Soup · without Synthetic",
    canonical_corpora: AF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "paper", paper_variant: "Soup({𝒜, ℱ})", avg_f1: null,
    merge_method: "Linear (uniform), 𝒮 removed",
    merge_components: ["sraj/CMB_MARK_CX_LRD", "sraj/CMB_FWEdu_V2_FastTxt_CX_LRD"],
    evidence: ["mergekit"],
    notes: "Figure 2. Excluding 𝒮 costs about 1.5 average F1 and improves some individual benchmarks.",
  },
  {
    hf_id: "sraj/Merge_Drop_FWebEduv2",
    display_name: "Drop-one Soup · without Climate Web",
    canonical_corpora: AS, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "paper", paper_variant: "Soup({𝒜, 𝒮})", avg_f1: null,
    merge_method: "Linear (uniform), ℱ removed",
    merge_components: ["sraj/CMB_MARK_CX_LRD", "sraj/CMB_WX_SYN_CX_LRD"],
    evidence: ["mergekit"],
    notes:
      "Figure 2. Excluding ℱ costs about 1.5 average F1. Because ℱ is the dropped corpus, this merge is unaffected by the FastText filtering question.",
  },

  // Merge variants released without a corresponding manuscript result
  {
    hf_id: "sraj/Merge_Linear_A2x",
    display_name: "Soup variant · Academic ×2",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "experimental", paper_variant: null, avg_f1: null,
    merge_method: "Linear, weights 𝒜 2.0 · ℱ 1.0 · 𝒮 1.0",
    merge_components: SOUP_COMPONENTS,
    evidence: ["collection", "mergekit"],
    notes: "Academic-upweighted Soup. Released, but no result is reported in the manuscript.",
  },
  {
    hf_id: "sraj/Merge_Linear_A3x",
    display_name: "Soup variant · Academic ×3",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "experimental", paper_variant: null, avg_f1: null,
    merge_method: "Linear, weights 𝒜 3.0 · ℱ 1.0 · 𝒮 1.0",
    merge_components: SOUP_COMPONENTS,
    evidence: ["collection", "mergekit"],
    notes: "Academic-upweighted Soup. Released, but no result is reported in the manuscript.",
  },
  {
    hf_id: "sraj/Merge_Linear_Fhalf",
    display_name: "Soup variant · Climate Web ×0.5",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "experimental", paper_variant: null, avg_f1: null,
    merge_method: "Linear, weights 𝒜 1.0 · ℱ 0.5 · 𝒮 1.0",
    merge_components: SOUP_COMPONENTS,
    evidence: ["collection", "mergekit"],
    notes:
      "Appendix F names F-half merging as one of three strategies examined, but Table 10 reports only θSoup, θTA(0.5) and θNorm. No published score.",
  },
  {
    hf_id: "sraj/Merge_Linear_Fhalf_CX_only",
    display_name: "Soup variant · Climate Web ×0.5 (Phase-1 components)",
    canonical_corpora: ASF, stage: "phase1", legacy_stage: "CX",
    family: "merged", status: "experimental", paper_variant: null, avg_f1: null,
    merge_method: "Linear, weights 𝒜 1.0 · ℱ 0.5 · 𝒮 1.0",
    merge_components: ["sraj/CMB_MARK_CX", "sraj/CMB_FWEdu_V2_FastTxt_CX", "sraj/CMB_WX_SYN_CX"],
    evidence: ["collection", "mergekit"],
    notes: "No published score; see Merge_Linear_Fhalf.",
  },
  {
    hf_id: "sraj/Merge_Linear_orig",
    display_name: "Soup · pre-FastText web component",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "superseded", paper_variant: null, avg_f1: null,
    merge_method: "Linear (uniform weight averaging, normalized)",
    merge_components: LEGACY_MERGE_COMPONENTS,
    superseded_by: "sraj/Merge_Linear",
    evidence: ["collection", "mergekit"],
    notes:
      "Identical merge recipe to Merge_Linear but built on CMB_FWEdu_V2_CX_LRD, the web checkpoint from before FastText filtering. Superseded by Merge_Linear, which uses the paper's ℱ.",
  },
  {
    hf_id: "sraj/Merge_Drop_MARK",
    display_name: "Drop-one Soup · without Academic (pre-FastText)",
    canonical_corpora: SF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "superseded", paper_variant: null, avg_f1: null,
    merge_method: "Linear (uniform), 𝒜 removed",
    merge_components: ["sraj/CMB_FWEdu_V2_CX_LRD", "sraj/CMB_WX_SYN_CX_LRD"],
    superseded_by: "sraj/Merge_Drop_MARK_FastText",
    evidence: ["mergekit"],
    notes: "Pre-FastText web component.",
  },
  {
    hf_id: "sraj/Merge_Drop_MARK_new_syn",
    display_name: "Drop-one Soup · without Academic (duplicate)",
    canonical_corpora: SF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "superseded", paper_variant: null, avg_f1: null,
    merge_method: "Linear (uniform), 𝒜 removed",
    merge_components: ["sraj/CMB_FWEdu_V2_CX_LRD", "sraj/CMB_WX_SYN_CX_LRD"],
    superseded_by: "sraj/Merge_Drop_MARK_FastText",
    duplicate_of: "sraj/Merge_Drop_MARK",
    evidence: ["mergekit"],
    notes:
      "The name suggests a newer synthetic component, but the mergekit config and the model.safetensors checksum are identical to Merge_Drop_MARK. Byte-for-byte the same weights.",
  },
  {
    hf_id: "sraj/Merge_Drop_MARK_add_new_syn",
    display_name: "Drop-one Soup · without Academic (duplicate)",
    canonical_corpora: SF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "superseded", paper_variant: null, avg_f1: null,
    merge_method: "Linear (uniform), 𝒜 removed",
    merge_components: ["sraj/CMB_FWEdu_V2_CX_LRD", "sraj/CMB_WX_SYN_CX_LRD"],
    superseded_by: "sraj/Merge_Drop_MARK_FastText",
    duplicate_of: "sraj/Merge_Drop_MARK",
    evidence: ["mergekit"],
    notes: "Byte-identical to Merge_Drop_MARK (same model.safetensors checksum).",
  },
  {
    hf_id: "sraj/Merge_Drop_SYN",
    display_name: "Drop-one Soup · without Synthetic (pre-FastText)",
    canonical_corpora: AF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "superseded", paper_variant: null, avg_f1: null,
    merge_method: "Linear (uniform), 𝒮 removed",
    merge_components: ["sraj/CMB_MARK_CX_LRD", "sraj/CMB_FWEdu_V2_CX_LRD"],
    superseded_by: "sraj/Merge_Drop_SYN_FastText",
    evidence: ["mergekit"],
    notes: "Pre-FastText web component.",
  },
  {
    hf_id: "sraj/Merge_Drop_FWebEduv2_new_syn",
    display_name: "Drop-one Soup · without Climate Web (duplicate)",
    canonical_corpora: AS, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "superseded", paper_variant: null, avg_f1: null,
    merge_method: "Linear (uniform), ℱ removed",
    merge_components: ["sraj/CMB_MARK_CX_LRD", "sraj/CMB_WX_SYN_CX_LRD"],
    duplicate_of: "sraj/Merge_Drop_FWebEduv2",
    evidence: ["mergekit"],
    notes: "Byte-identical to Merge_Drop_FWebEduv2 (same model.safetensors checksum).",
  },
  {
    hf_id: "sraj/Merge_Drop_FWebEduv2_add_new_syn",
    display_name: "Drop-one Soup · without Climate Web (duplicate)",
    canonical_corpora: AS, stage: "phase2", legacy_stage: "CX_LRD",
    family: "merged", status: "superseded", paper_variant: null, avg_f1: null,
    merge_method: "Linear (uniform), ℱ removed",
    merge_components: ["sraj/CMB_MARK_CX_LRD", "sraj/CMB_WX_SYN_CX_LRD"],
    duplicate_of: "sraj/Merge_Drop_FWebEduv2",
    evidence: ["mergekit"],
    notes: "Byte-identical to Merge_Drop_FWebEduv2 (same model.safetensors checksum).",
  },

  // ─────────────── Jointly trained: Phase 2 (CX + LRD) ───────────────
  {
    hf_id: "sraj/CMB_MARK_CX_LRD",
    display_name: "ClimateModernBERT · Academic",
    canonical_corpora: A, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "paper", paper_variant: "θ{𝒜}", avg_f1: 74.4,
    evidence: ["collection", "mergekit"],
    notes:
      "The 𝒜 single-source Phase-2 checkpoint, and the academic component of every released merge — confirmed from the merge configs.",
  },
  {
    hf_id: "sraj/CMB_FWEdu_V2_FastTxt_CX_LRD",
    display_name: "ClimateModernBERT · Climate Web",
    canonical_corpora: F, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "paper", paper_variant: "θ{ℱ}", avg_f1: 74.5,
    evidence: ["collection", "mergekit"],
    notes:
      "The ℱ single-source Phase-2 checkpoint. FastTxt marks the FastText climate-filtering stage, which is how the paper's ℱ is defined.",
  },
  {
    hf_id: "sraj/CMB_WX_SYN_CX_LRD",
    display_name: "ClimateModernBERT · Synthetic",
    canonical_corpora: S, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "paper", paper_variant: "θ{𝒮}", avg_f1: 74.5,
    evidence: ["mergekit"],
    notes:
      "The paper's 𝒮 Phase-2 checkpoint, confirmed by an author. It is the verified synthetic component of θSoup and every other released merge, which is what §5.2 describes Table 4 as merging.",
  },
  {
    hf_id: "sraj/CMB_SYN_CX_LRD",
    display_name: "Synthetic corpus variant · Phase 2",
    canonical_corpora: S, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "experimental", paper_variant: null, avg_f1: null,
    evidence: ["collection"],
    notes:
      "A later synthetic-corpus run, listed as 𝒮 in the curated collection but not the synthetic component of any released merge. An author confirmed CMB_WX_SYN_CX_LRD as the paper's θ𝒮, so no manuscript score is attributed here.",
  },
  {
    hf_id: "sraj/CMB_MARK_WX_SYN_CX_LRD",
    display_name: "ClimateModernBERT · Academic + Synthetic",
    canonical_corpora: AS, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "paper", paper_variant: "θ{𝒜, 𝒮}", avg_f1: 74.5,
    evidence: ["collection", "name"],
    notes:
      "MARK-only, but no CMB_MARK_WX_SYN_ZYDA_CX_LRD exists on the Hub, so the MARK/ZYDA supersession rule does not apply here. This is the checkpoint the curated collection designates for {𝒜, 𝒮}.",
  },
  {
    hf_id: "sraj/CMB_FWEdu_V2_FastTxt_MARK_CX_LRD",
    display_name: "ClimateModernBERT · Academic + Climate Web",
    canonical_corpora: AF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "paper", paper_variant: "θ{𝒜, ℱ}", avg_f1: 74.7,
    evidence: ["collection", "name"],
    notes: "Joint continued pretraining on the union of 𝒜 and ℱ.",
  },
  {
    hf_id: "sraj/CMB_FWEdu_V2_FastTxt_SYN_CX_LRD",
    display_name: "ClimateModernBERT · Synthetic + Climate Web",
    canonical_corpora: SF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "paper", paper_variant: "θ{𝒮, ℱ}", avg_f1: 74.1,
    evidence: ["collection", "name"],
    notes: "Joint continued pretraining on the union of 𝒮 and ℱ.",
  },
  {
    hf_id: "sraj/CMB_FWEdu_V2_FastTxt_MARK_SYN_CX_LRD",
    display_name: "ClimateModernBERT · Academic + Synthetic + Climate Web",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "paper", paper_variant: "θ{𝒜, 𝒮, ℱ}", avg_f1: 74.8,
    evidence: ["collection", "name"],
    notes:
      "Joint training on the full corpus union — the direct comparison point for θSoup, which reaches 76.3 on the same effective data.",
  },

  // ─────────────── Jointly trained: Phase 1 (CX only) ───────────────
  {
    hf_id: "sraj/CMB_MARK_CX",
    display_name: "ClimateModernBERT · Academic · Phase 1",
    canonical_corpora: A, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "recommended", paper_variant: "θ{𝒜}", avg_f1: 75.3,
    evidence: ["mergekit", "name"],
    notes:
      "The strongest jointly trained configuration in the manuscript (75.3 average F1) and the best single non-merged checkpoint. Confirmed as the academic component of the Phase-1 merges.",
  },
  {
    hf_id: "sraj/CMB_FWEdu_V2_FastTxt_CX",
    display_name: "ClimateModernBERT · Climate Web · Phase 1",
    canonical_corpora: F, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "paper", paper_variant: "θ{ℱ}", avg_f1: 74.1,
    evidence: ["mergekit", "name"],
    notes: "Confirmed as the web component of the Phase-1 merges.",
  },
  {
    hf_id: "sraj/CMB_WX_SYN_CX",
    display_name: "ClimateModernBERT · Synthetic · Phase 1",
    canonical_corpora: S, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "paper", paper_variant: "θ{𝒮}", avg_f1: 73.4,
    evidence: ["mergekit"],
    notes:
      "The paper's 𝒮 Phase-1 checkpoint, and the verified synthetic component of the Phase-1 merges. Confirmed by an author alongside its Phase-2 counterpart.",
  },
  {
    hf_id: "sraj/CMB_SYN_CX",
    display_name: "Synthetic corpus variant · Phase 1",
    canonical_corpora: S, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "experimental", paper_variant: null, avg_f1: null,
    evidence: ["name"],
    notes: "Phase-1 counterpart of CMB_SYN_CX_LRD; not the paper's θ𝒮. See CMB_WX_SYN_CX.",
  },
  {
    hf_id: "sraj/CMB_MARK_WX_SYN_ZYDA_CX",
    display_name: "ClimateModernBERT · Academic + Synthetic · Phase 1",
    canonical_corpora: AS, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "paper", paper_variant: "θ{𝒜, 𝒮}", avg_f1: 74.8,
    evidence: ["name"],
    notes:
      "Preferred over CMB_MARK_WX_SYN_CX: MARK and ZYDA are both academic components, so the MARK+ZYDA run is the more complete academic configuration.",
  },
  {
    hf_id: "sraj/CMB_FWEdu_V2_FastTxt_MARK_CX",
    display_name: "ClimateModernBERT · Academic + Climate Web · Phase 1",
    canonical_corpora: AF, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "paper", paper_variant: "θ{𝒜, ℱ}", avg_f1: 74.3,
    evidence: ["name"],
    notes: "Phase-1 counterpart of CMB_FWEdu_V2_FastTxt_MARK_CX_LRD.",
  },

  // ─────────────── Legacy / experimental ───────────────
  {
    hf_id: "sraj/CMB_MARK_WX_SYN_CX",
    display_name: "Academic + Synthetic · Phase 1 (MARK only)",
    canonical_corpora: AS, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "superseded", paper_variant: null, avg_f1: null,
    superseded_by: "sraj/CMB_MARK_WX_SYN_ZYDA_CX",
    evidence: ["name"],
    notes: "Uses only the MARK academic component; the MARK+ZYDA run covers a more complete 𝒜.",
  },
  {
    hf_id: "sraj/CMB_FWEdu_V2_MARK_WX_SYN_CX",
    display_name: "All three corpora · Phase 1 (MARK only, pre-FastText)",
    canonical_corpora: ASF, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "superseded", paper_variant: null, avg_f1: null,
    superseded_by: "sraj/CMB_FWEdu_V2_MARK_WX_SYN_ZYDA_CX",
    evidence: ["name"],
    notes: "MARK-only academic component and pre-FastText web corpus.",
  },
  {
    hf_id: "sraj/CMB_FWEdu_V2_MARK_WX_SYN_CX_LRD",
    display_name: "All three corpora · Phase 2 (MARK only, pre-FastText)",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "superseded", paper_variant: null, avg_f1: null,
    superseded_by: "sraj/CMB_FWEdu_V2_MARK_WX_SYN_ZYDA_CX_LRD",
    evidence: ["name"],
    notes: "MARK-only academic component and pre-FastText web corpus.",
  },
  {
    hf_id: "sraj/CMB_FWEdu_V2_MARK_WX_SYN_ZYDA_CX",
    display_name: "All three corpora · Phase 1 (pre-FastText web)",
    canonical_corpora: ASF, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "legacy", paper_variant: null, avg_f1: null,
    evidence: ["name"],
    notes:
      "MARK+ZYDA academic component, but the web corpus predates FastText filtering, so this is not the paper's {𝒜, 𝒮, ℱ}.",
  },
  {
    hf_id: "sraj/CMB_FWEdu_V2_MARK_WX_SYN_ZYDA_CX_LRD",
    display_name: "All three corpora · Phase 2 (pre-FastText web)",
    canonical_corpora: ASF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "legacy", paper_variant: null, avg_f1: null,
    evidence: ["name"],
    notes: "As above, with LRD specialization.",
  },
  {
    hf_id: "sraj/CMB_FWEdu_V2_WX_SYN_CX_LRD",
    display_name: "Synthetic + Climate Web · Phase 2 (pre-FastText)",
    canonical_corpora: SF, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "legacy", paper_variant: null, avg_f1: null,
    evidence: ["name"],
    notes: "Pre-FastText web corpus; superseded in the paper's setup by CMB_FWEdu_V2_FastTxt_SYN_CX_LRD.",
  },
  {
    hf_id: "sraj/CMB_WX_SYN_ZYDA_CX",
    display_name: "Academic (ZYDA) + Synthetic · Phase 1",
    canonical_corpora: AS, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "legacy", paper_variant: null, avg_f1: null,
    evidence: ["name"], ambiguous: true,
    notes:
      "Academic component is ZYDA without MARK — the mirror image of the usual pairing. Not part of the MARK/ZYDA supersession rule and not matched to a manuscript configuration.",
  },
  {
    hf_id: "sraj/CMB_FWEdu_V2_CX",
    display_name: "Climate Web · Phase 1 (pre-FastText)",
    canonical_corpora: F, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "legacy", paper_variant: null, avg_f1: null,
    evidence: ["name"],
    notes: "FineWeb-Edu before the FastText climate filter.",
  },
  {
    hf_id: "sraj/CMB_FWEdu_V2_CX_LRD",
    display_name: "Climate Web · Phase 2 (pre-FastText)",
    canonical_corpora: F, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "legacy", paper_variant: null, avg_f1: null,
    evidence: ["collection", "mergekit"],
    notes:
      "FineWeb-Edu before the FastText climate filter. Kept in the curated collection because it is the web component of Merge_Linear_orig and of the TA / TIES / DARE merges.",
  },
  {
    hf_id: "sraj/CMB_SYN_QWEN35_122B_FP8_10K_SEED42_CX_LRD",
    display_name: "Synthetic generator study · Qwen3.5-122B · Phase 2",
    canonical_corpora: S, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "experimental", paper_variant: null, avg_f1: null,
    synthetic_generator: "Qwen3.5-122B-A10B",
    evidence: ["name"],
    notes:
      "Labelled with the generator the manuscript describes for 𝒮 (§3.1), but its weights differ from the confirmed θ𝒮. A generation experiment, not the paper's synthetic corpus.",
  },
  {
    hf_id: "sraj/CMB_SYN_QWEN35_122B_FP8_10K_SEED42_CX",
    display_name: "Synthetic generator study · Qwen3.5-122B · Phase 1",
    canonical_corpora: S, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "experimental", paper_variant: null, avg_f1: null,
    synthetic_generator: "Qwen3.5-122B-A10B",
    evidence: ["name"],
    notes: "Phase-1 counterpart of the above.",
  },
  {
    hf_id: "sraj/CMB_SYN_QWEN3_30B_A3B_FP8_10K_SEED42_CX_LRD",
    display_name: "Synthetic generator study · Qwen3-30B-A3B · Phase 2",
    canonical_corpora: S, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "experimental", paper_variant: null, avg_f1: null,
    synthetic_generator: "Qwen3-30B-A3B",
    evidence: ["name"],
    notes: "Synthetic-data generation experiment with a smaller generator. Not the manuscript's 𝒮.",
  },
  {
    hf_id: "sraj/CMB_SYN_QWEN3_30B_A3B_FP8_10K_SEED42_CX",
    display_name: "Synthetic generator study · Qwen3-30B-A3B · Phase 1",
    canonical_corpora: S, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "experimental", paper_variant: null, avg_f1: null,
    synthetic_generator: "Qwen3-30B-A3B",
    evidence: ["name"],
    notes: "Phase-1 counterpart of the above.",
  },
  {
    hf_id: "sraj/CMB_WX_CX_LRD",
    display_name: "WX data shard · Phase 2",
    canonical_corpora: [], stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "experimental", paper_variant: null, avg_f1: null,
    evidence: ["name"], ambiguous: true,
    notes:
      "Unverified. The WX token appears elsewhere only as part of WX_SYN; on its own it does not map to 𝒜, ℱ or 𝒮 with any confidence.",
  },
  {
    hf_id: "sraj/CMB_WX_CX",
    display_name: "WX data shard · Phase 1",
    canonical_corpora: [], stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "experimental", paper_variant: null, avg_f1: null,
    evidence: ["name"], ambiguous: true,
    notes: "Unverified; see CMB_WX_CX_LRD.",
  },
  {
    hf_id: "sraj/CMB_MICH_FWEdu_CX",
    display_name: "Early web pilot · Phase 1",
    canonical_corpora: F, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "legacy", paper_variant: null, avg_f1: null,
    evidence: ["name"],
    notes: "Exploratory run predating the corpus definitions in the manuscript.",
  },
  {
    hf_id: "sraj/CMB_MICH_FWEdu_DEDUP_CX",
    display_name: "Early web pilot · deduplicated · Phase 1",
    canonical_corpora: F, stage: "phase1", legacy_stage: "CX",
    family: "joint", status: "legacy", paper_variant: null, avg_f1: null,
    evidence: ["name"],
    notes: "Exploratory run predating the corpus definitions in the manuscript.",
  },
  {
    hf_id: "sraj/CMB_MICH_FWEdu_DEDUP_CX_LRD",
    display_name: "Early web pilot · deduplicated · Phase 2",
    canonical_corpora: F, stage: "phase2", legacy_stage: "CX_LRD",
    family: "joint", status: "legacy", paper_variant: null, avg_f1: null,
    evidence: ["name"],
    notes: "Exploratory run predating the corpus definitions in the manuscript.",
  },
];


/* ───────────────────── Organization migration ─────────────────────
 * The paper-relevant checkpoints are being republished under the
 * CMB-ClimateModernBERT organization with names that read as the paper's own
 * notation: corpora in A_S_F order, then the training stage.
 *
 *   A_S_F_CX      Phase 1 on academic + synthetic + web
 *   A_S_F_CX_LRD  the same, plus Phase 2 LRD specialization
 *   Merge_*       parameter-space merges, suffixed by the components' stage
 *
 * The original sraj/* repositories are NOT touched — this account has no write
 * access to them, so every legacy link keeps working.
 *
 * θ𝒮 is CMB_WX_SYN_CX_LRD, confirmed by an author: it is the synthetic
 * component of every released merge, which is what §5.2 describes Table 4 as
 * merging. See docs/model-naming.md.
 */
export const HF_ORG = "CMB-ClimateModernBERT";

/** Legacy sraj id → new repo name within HF_ORG. Absent = not migrated. */
export const migration: Record<string, string> = {
  // Jointly trained · Phase 2 (CX + LRD)
  "sraj/CMB_MARK_CX_LRD": "A_CX_LRD",
  "sraj/CMB_WX_SYN_CX_LRD": "S_CX_LRD",
  "sraj/CMB_FWEdu_V2_FastTxt_CX_LRD": "F_CX_LRD",
  "sraj/CMB_MARK_WX_SYN_CX_LRD": "A_S_CX_LRD",
  "sraj/CMB_FWEdu_V2_FastTxt_MARK_CX_LRD": "A_F_CX_LRD",
  "sraj/CMB_FWEdu_V2_FastTxt_SYN_CX_LRD": "S_F_CX_LRD",
  "sraj/CMB_FWEdu_V2_FastTxt_MARK_SYN_CX_LRD": "A_S_F_CX_LRD",

  // Jointly trained · Phase 1 (CX only). {S,F} and {A,S,F} were never published.
  "sraj/CMB_MARK_CX": "A_CX",
  "sraj/CMB_WX_SYN_CX": "S_CX",
  "sraj/CMB_FWEdu_V2_FastTxt_CX": "F_CX",
  "sraj/CMB_MARK_WX_SYN_ZYDA_CX": "A_S_CX",
  "sraj/CMB_FWEdu_V2_FastTxt_MARK_CX": "A_F_CX",

  // Parameter-space merges of the three Phase-2 components (Table 4)
  "sraj/Merge_Linear": "Merge_Soup_LRD",
  "sraj/TA_Lambda10_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD": "Merge_TA_L10_LRD",
  "sraj/TA_Lambda05_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD": "Merge_TA_L05_LRD",
  "sraj/TIES_D07_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD": "Merge_TIES_D07_LRD",
  "sraj/TIES_D05_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD": "Merge_TIES_D05_LRD",
  "sraj/DARE_TIES_D05_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD": "Merge_DARE_D05_LRD",
  "sraj/DARE_TIES_D07_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD": "Merge_DARE_D07_LRD",
  "sraj/Merge_Linear_NormBalanced": "Merge_Norm_LRD",

  // Appendix F — merges built from Phase-1 components
  "sraj/Merge_Linear_CX_only": "Merge_Soup_CX",
  "sraj/Merge_Linear_NormBalanced_CX_only": "Merge_Norm_CX",

  // Figure 2 — drop-one Soup ablations
  "sraj/Merge_Drop_MARK_FastText": "Merge_Soup_drop_A_LRD",
  "sraj/Merge_Drop_SYN_FastText": "Merge_Soup_drop_S_LRD",
  "sraj/Merge_Drop_FWebEduv2": "Merge_Soup_drop_F_LRD",
};

/** Full new id, or null when the checkpoint stays only under sraj/*. */
export function newId(hfId: string): string | null {
  const name = migration[hfId];
  return name ? `${HF_ORG}/${name}` : null;
}

/** What the site should show as the model's primary id. */
export function primaryId(m: { hf_id: string }): string {
  return newId(m.hf_id) ?? m.hf_id;
}

/* ───────────────────────── Derived helpers ───────────────────────── */

export const hfUrl = (id: string) => `https://huggingface.co/${id}`;

export const repoName = (id: string) => id.split("/")[1] ?? id;

/** "ClimateModernBERT · Academic + Synthetic · Phase 1" */
export function longName(m: ModelEntry): string {
  const name = m.display_name;
  // Several legacy entries already name their stage (often with a qualifier,
  // e.g. "… · Phase 1 (MARK only)"), so only append it when it is absent.
  if (m.family === "merged" || m.canonical_corpora.length === 0) return name;
  if (name.includes(stageShort[m.stage])) return name;
  // NBSP before the separator so it never begins a wrapped line.
  return `${name}\u00a0· ${stageShort[m.stage]}`;
}

export function notation(m: ModelEntry): string {
  return m.canonical_corpora.length ? paperNotation(m.canonical_corpora) : "—";
}

export function corpusText(m: ModelEntry): string {
  return m.canonical_corpora.length ? corpusPhrase(m.canonical_corpora) : "Unverified";
}

export const statusLabel: Record<Status, string> = {
  recommended: "Recommended",
  paper: "Paper",
  experimental: "Experimental",
  legacy: "Legacy",
  superseded: "Superseded",
};

/** Order used by the explorer: strongest evidence and clearest provenance first. */
const statusRank: Record<Status, number> = {
  recommended: 0, paper: 1, experimental: 2, legacy: 3, superseded: 4,
};

export const sortedModels = [...models].sort((a, b) => {
  const s = statusRank[a.status] - statusRank[b.status];
  if (s !== 0) return s;
  if (a.family !== b.family) return a.family === "merged" ? -1 : 1;
  return (b.avg_f1 ?? -1) - (a.avg_f1 ?? -1);
});

export const counts = {
  total: models.length,
  paper: models.filter((m) => m.status === "paper" || m.status === "recommended").length,
  merged: models.filter((m) => m.family === "merged").length,
  legacy: models.filter((m) => m.status === "legacy" || m.status === "superseded").length,
  ambiguous: models.filter((m) => m.ambiguous).length,
  superseded: models.filter((m) => m.superseded_by).length,
};
