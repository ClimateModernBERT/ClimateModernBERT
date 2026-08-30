# Model naming: historical checkpoints vs. paper notation

> **The paper's checkpoints now live under
> [`CMB-ClimateModernBERT`](https://huggingface.co/CMB-ClimateModernBERT)** with names that read
> as the paper's own notation. The original `sraj/*` repositories are untouched and every
> existing link still works. See [The republished naming scheme](#the-republished-naming-scheme).

The Hugging Face repository names for ClimateModernBERT are **historical experiment
identifiers**. They record which internal data shard a training run consumed at the
moment it was launched, and they accumulated over roughly seven months of
experimentation. The manuscript, written afterwards, uses a much smaller and cleaner
vocabulary.

The project website and `docs/model-inventory.md` present the paper's notation and show
the raw repository id underneath. This document is the mapping between the two.

## The republished naming scheme

Twenty-five checkpoints — every configuration with a number in the manuscript — are published
under the `CMB-ClimateModernBERT` organization. The name is the corpus set in `A_S_F` order,
then the training stage:

| Pattern | Meaning | Example |
|---|---|---|
| `A_CX` | Phase 1 on academic only | `A_CX` — 75.3 avg F1, best single checkpoint |
| `A_S_F_CX_LRD` | Phase 1 + Phase 2 on all three corpora | `A_S_F_CX_LRD` — 74.8 |
| `Merge_Soup_LRD` | uniform weight average of the Phase-2 components | `Merge_Soup_LRD` — 76.3, recommended |
| `Merge_TIES_D07_LRD` | TIES merge, drop ratio 0.7, Phase-2 components | |
| `Merge_Soup_CX` | the same merge built from Phase-1 components | |
| `Merge_Soup_drop_A_LRD` | drop-one Soup ablation, academic removed | |

Full jointly trained set: `A_CX`, `S_CX`, `F_CX`, `A_S_CX`, `A_F_CX` and their `_CX_LRD`
counterparts, plus `S_F_CX_LRD` and `A_S_F_CX_LRD`. The {𝒮, ℱ} and {𝒜, 𝒮, ℱ} Phase-1
checkpoints were never published, so they have no republished name either.

The complete legacy → new mapping is in
[`site/src/data/models.ts`](../site/src/data/models.ts) (`migration`) and in
[`huggingface/manifests/models.json`](../huggingface/manifests/models.json) (`new_id`).
`scripts/migrate_to_org.py` performs the republish; it is idempotent and never writes to
`sraj/*`.

## The paper's vocabulary

There are exactly **three** corpora:

| Symbol | Name | Tokens | What it is |
|---|---|---|---|
| 𝒜 | Academic | ~1.28B | Peer-reviewed journal articles, the ClimateNews archive 2000–2022, climate arXiv preprints, climate handbooks |
| ℱ | Climate Web | ~5B | FineWeb-Edu filtered for climate relevance |
| 𝒮 | Synthetic | ~0.14B | LLM-generated climate text conditioned on in-domain seed excerpts |

and exactly **two** training stages:

| Stage | Paper term | Legacy suffix |
|---|---|---|
| Phase 1 | Continued pretraining (context extension) | `CX` |
| Phase 2 | LRD specialization, applied after Phase 1 | `CX_LRD` |

So `CX` = Phase 1 only, and `CX_LRD` = Phase 1 followed by Phase 2. The Phase-2
checkpoints are the paper's primary models; Phase-1 checkpoints are retained as an
ablation isolating what the learning-rate-decay stage contributes.

## Token-by-token mapping

| Historical token | Maps to | Meaning |
|---|---|---|
| `MARK` | 𝒜 | An academic-source data component. One of two internal academic shards. |
| `ZYDA` | 𝒜 | A second academic-source data component. |
| `SYN` | 𝒮 | Synthetic climate data. |
| `WX_SYN` | 𝒮 | Synthetic climate data under an earlier naming stage. |
| `FWEdu_V2` | ℱ | FineWeb-Edu-derived web corpus, **before** the FastText filtering stage. |
| `FWEdu_V2_FastTxt` | ℱ | FineWeb-Edu after FastText climate filtering. This is the paper's ℱ. |
| `QWEN35_122B`, `QWEN3_30B_A3B` | — | Synthetic-data *generation* experiments, named for the generator model. |
| `MICH` | — | Early exploratory shard, predating the corpus definitions in the manuscript. |
| `CX` | Phase 1 | Context-extension continued pretraining. |
| `CX_LRD` | Phase 2 | Phase 1 followed by learning-rate-decay specialization. |

Two things this table is deliberately careful about:

- **ZYDA is not a fourth corpus.** `MARK` and `ZYDA` are both internal components used
  to construct 𝒜. The paper has no notion of them.
- **`FastTxt` is not a fourth corpus either.** It is the second stage of ℱ's filtering
  pipeline (a keyword filter, then a FastText classifier). A `FastTxt` repo and a
  non-`FastTxt` repo differ in *how much of FineWeb-Edu survived filtering*, not in
  which corpus was used.

A machine-readable version of this mapping, plus a parser that applies it, lives in
[`site/src/data/naming.ts`](../site/src/data/naming.ts). The parser is used to
cross-check every hand-written entry in the manifest; both agree on all 56 checkpoints.

## The MARK / ZYDA supersession rule

Some checkpoints differ only in that an earlier run used `MARK` where a later, more
complete run used `MARK` + `ZYDA`. Since both are academic components, the MARK+ZYDA
run is the more complete 𝒜 configuration and supersedes the MARK-only one.

The rule applies **only when the MARK+ZYDA counterpart actually exists on the Hub.**
Verified pairs:

| Superseded (MARK only) | Superseded by (MARK + ZYDA) |
|---|---|
| `CMB_MARK_WX_SYN_CX` | `CMB_MARK_WX_SYN_ZYDA_CX` |
| `CMB_FWEdu_V2_MARK_WX_SYN_CX` | `CMB_FWEdu_V2_MARK_WX_SYN_ZYDA_CX` |
| `CMB_FWEdu_V2_MARK_WX_SYN_CX_LRD` | `CMB_FWEdu_V2_MARK_WX_SYN_ZYDA_CX_LRD` |

**Not superseded, despite the name pattern:**

- `CMB_MARK_WX_SYN_CX_LRD` — no `CMB_MARK_WX_SYN_ZYDA_CX_LRD` exists on the Hub, so
  there is nothing to supersede it. It is the checkpoint the curated collection
  designates for the paper's {𝒜, 𝒮} Phase-2 configuration, and it stays in the
  primary catalog.
- `CMB_MARK_CX` and `CMB_MARK_CX_LRD` — single-source 𝒜 checkpoints with no ZYDA
  counterpart. `CMB_MARK_CX` is the strongest jointly trained model in the paper.

Superseded checkpoints are excluded from the site's primary catalog but remain listed,
linked, and reachable. **Nothing has been deleted from the Hugging Face Hub.**

## Byte-identical duplicates

Three pairs of repositories carry names implying a newer synthetic component but hold
the *same weights* — verified by comparing the `model.safetensors` SHA-256 returned by
the Hub, and by reading their `mergekit_config.yml`:

| Repositories | Relationship |
|---|---|
| `Merge_Drop_FWebEduv2`, `Merge_Drop_FWebEduv2_new_syn`, `Merge_Drop_FWebEduv2_add_new_syn` | Identical checksum; identical merge config |
| `Merge_Drop_MARK`, `Merge_Drop_MARK_new_syn`, `Merge_Drop_MARK_add_new_syn` | Identical checksum; identical merge config |

The `_new_syn` / `_add_new_syn` suffixes appear to record an intent that was never
carried through.

## Open questions for the authors

These could not be resolved from the manuscript, the curated collection, and the merge
configurations. They are flagged in the model explorer as *needs confirmation* rather
than guessed at.

### 1. ~~Which repository is the paper's θ𝒮?~~ — resolved

**Answered by an author: `CMB_WX_SYN_CX_LRD`**, republished as
[`CMB-ClimateModernBERT/S_CX_LRD`](https://huggingface.co/CMB-ClimateModernBERT/S_CX_LRD)
(and `CMB_WX_SYN_CX` → `S_CX` at Phase 1).

Two repositories were candidates, with different `model.safetensors` checksums.
`CMB_WX_SYN_CX_LRD` is the verified synthetic component of θ<sub>Soup</sub> and of every other
released merge, which is what §5.2 describes Table 4 as merging. `CMB_SYN_CX_LRD` — listed as
𝒮 in the curated collection but named in no merge config — is now catalogued as a later
synthetic-corpus variant with no manuscript score attributed to it, as is
`CMB_SYN_QWEN35_122B_FP8_10K_SEED42_CX_LRD`.

### 2. Which ℱ component do the TA / TIES / DARE merges use?

`Merge_Linear` (θ<sub>Soup</sub>) merges `CMB_FWEdu_V2_FastTxt_CX_LRD` — the paper's ℱ.
The six `TA_*`, `TIES_*` and `DARE_TIES_*` repositories instead name
`CMB_FWEdu_V2_CX_LRD`, the **pre-FastText** web checkpoint, in both their repo names and
their `mergekit_config.yml`.

Table 4 presents all seven merges as operating over the same effective corpus, so either
the released merge artifacts were built from an earlier component set than the reported
numbers, or the numbers correspond to these components. Their correspondence to Table 4
is recorded as "by merge method and hyperparameter, not by verified component set".

### 3. Two Phase-1 checkpoints are not published

Table 2 reports Phase-1 results for {𝒮, ℱ} and {𝒜, 𝒮, ℱ}, but no
`CMB_FWEdu_V2_FastTxt_SYN_CX` or `CMB_FWEdu_V2_FastTxt_MARK_SYN_CX` exists on the Hub.
Their Phase-2 counterparts are both published.

### 4. `CMB_WX_CX` / `CMB_WX_CX_LRD`

The `WX` token appears everywhere else only inside `WX_SYN`. On its own it does not map
to 𝒜, ℱ or 𝒮 with any confidence, so these two are catalogued as *Unverified*.

## Two transcription notes about the manuscript

Neither is acted on — the site reports the table values as printed — but both are worth
a look before the next revision:

1. **§5.2 body text vs. Table 4.** The text says θ<sub>Soup</sub> improves "by +1.7 over
   joint training on the same effective corpus (74.6)". Table 2 and Table 4 both print
   **74.8 ± 1.7** for θ<sup>LRD</sup><sub>{𝒜,𝒮,ℱ}</sub>. The site uses 74.8 and states
   the gap as +1.5.
2. **Table 2, Phase-1 block.** The {𝒮, ℱ} row is identical to the {ℱ} row on all nine
   per-task scores (85.4, 64.8, 94.5, 68.0, 76.3, 99.2, 59.1, 97.2, 23.4) while the
   averages differ (74.2 vs 74.1). The site reproduces both rows as printed.
