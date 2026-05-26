**"Climate-ModernBERT: Revisiting Corpus Composition for Domain-Adaptive Continued Pretraining."**

<div align="center">
    <img src="./pics/pipeline.png" alt="Link to PDF" height="auto" style="width:95%;">
</div>

## Repo layout

```
ClimateModernBERT/
├── continue_pretrain/        # Phase 1 + Phase 2 CPT (Composer / FlexBERT)
│   ├── context_ext.sh                                  # SLURM: launch Phase 1
│   ├── lr_decay.sh                                     # SLURM: launch Phase 2
│   ├── modernbert-base-context-extension.yaml          # Phase 1 hyperparams (§3.2)
│   ├── modernbert-base-learning-rate-decay.yaml        # Phase 2 hyperparams (§3.2)
│   ├── requirements.txt
│   └── data_pipeline/
│       ├── nemo_climate.sh                             # SLURM: dedup pipeline
│       ├── nemo_pipeline_climate.py                    # NeMo-Curator: clean → exact dedup → fuzzy dedup → JSONL
│       ├── run_fineweb_filter.sh                       # SLURM: launch FineWeb-Edu climate filter
│       └── stream_filter_upload_fineweb.py             # Stage-1 keyword filter + async HF Hub upload (§3.1, F)
│
├── src/
│   ├── data/                  # Academic-corpus prep (A)
│   │   ├── download_and_process_gdrive.py              # Pull raw journal exports
│   │   ├── extract_xml_text.py, extract_xml_for_dedup.py
│   │   ├── decontaminate.py, decontaminate_check.py, decontaminate_large.py
│   │   ├── deduplication.py, dedup_instructions.md     # Google's suffix-array dedup (see third_party/)
│   │   ├── process_all_datasets.py, create_final_dataset.sh
│   │   └── count_tokens.py
│   └── synthetic/             # Synthetic corpus (S) — Qwen3-30B via vLLM
│       ├── climatenew_vllm.py                          # Driver: seed → prompts → generation → JSONL
│       ├── chat.py, vllm_utils.py                      # Local OpenAI-compatible client + decoding configs
│       └── scripts/data_creation{,_1,_2}.sh            # Three SLURM jobs at seeds 42/43/44
│
├── model_merging/             # Post-hoc parameter-space merging (§4)
│   ├── merge.sh                                        # Driver: runs mergekit + pushes to HF Hub
│   └── merge_configs/                                  # 10 mergekit YAMLs (Soup, TA, TIES, DARE, drop-one)
│
├── eval/                      # Downstream fine-tuning + benchmark eval (§4.3, §5)
│   ├── config_updated.json                             # 9 tasks incl. ClimRetrieve
│   ├── config_without_GLUE.json                        # 8 tasks (no retrieval)
│   ├── multitask_finetuning_updated.py                 # n=3 seed fine-tuning
│   ├── multitask_finetuning_10seeds.py                 # n=10 paired-seed (synthetic ablation in §5.1)
│   ├── benchmark_evaluation_updated.py                 # Score saved checkpoints
│   ├── run_multitask.sh, run_multitask_pipeline.sh     # SLURM entry points
│   └── README.md                                       # Eval-side usage notes
│
├── utils/                     # Format conversions and HF Hub uploads
│   ├── convert_any_to_mds.py + run_hf_to_mds.sh        # JSONL/HF → MosaicML Streaming (MDS) shards for CPT
│   ├── convert_to_hf.py + convert_to_hf.sh             # Composer checkpoint → HF Transformers (FlexBERT)
│   ├── upload_to_hub.py + upload_to_hub.sh             # Push HF-format checkpoints to the Hub
│   └── count_tokens.py                                 # Rough token tally on an MDS dataset
│
└── third_party/
    └── deduplicate-text-datasets/                      # Google submodule used by src/data/deduplication.py
```

---

## How to reproduce, end-to-end

There are four stages. Each one can run independently if you bring your own intermediate artifacts.

### 1. Build the three corpora ($\mathcal{A}$, $\mathcal{F}$, $\mathcal{S}$)

**Academic ($\mathcal{A}$, ~1.28B tokens after dedup; §3.1, App. B).**
Raw journal XML/PDF lives outside this repo. Use `src/data/extract_xml_text.py` and the PDF helpers in `src/data/` to lift cleaned text, then run `src/data/deduplication.py` (which calls the Google suffix-array tool vendored under `third_party/deduplicate-text-datasets/`) followed by `src/data/decontaminate.py` against the nine downstream eval sets.

**Web ($\mathcal{F}$, ~5.1B tokens; §3.1, App. C).**
`continue_pretrain/data_pipeline/run_fineweb_filter.sh` streams FineWeb-Edu through the 112-keyword filter and a FastText classifier (threshold 0.5) and pushes survivors to an HF Hub dataset shard-by-shard. Edit `DATASET`, `HUB_REPO_ID`, and `SUBSET` at the top of the shell script.

**Synthetic ($\mathcal{S}$, ~0.14B tokens; §3.1, App. D).**
Seed excerpts (first 800 chars of each in-domain doc) are fed to Qwen3-30B via vLLM. The three style templates ("public awareness", "industry perspective", "environmental journalism") are hard-coded in `src/synthetic/climatenew_vllm.py`. `src/synthetic/scripts/data_creation{,_1,_2}.sh` are the same job at seeds 42 / 43 / 44 for variance.

All three corpora then pass through `continue_pretrain/data_pipeline/nemo_pipeline_climate.py` (text cleaning → exact MD5 dedup → fuzzy MinHash–LSH with 24-char n-grams, 20 bands, 13 hashes/band; §App. A). The output is a single JSONL per corpus.

### 2. Pack into MDS shards

`utils/convert_any_to_mds.py` is the one place you wire your JSONL paths and HF dataset refs into a single MosaicML Streaming dataset. The `data_local` field in the Composer YAMLs points at this output directory.

### 3. Two-stage continued pretraining (§3.2)

```bash
# Phase 1 — domain adaptation at constant LR=3e-4, 3 epochs, MLM 30%, batch 576, seqlen 8192.
sbatch continue_pretrain/context_ext.sh

# Phase 2 — LRD specialization (1 - sqrt(t) schedule, alpha_f=1e-3), 3 more epochs.
# Set `load_path` in the YAML to your Phase-1 checkpoint, then:
sbatch continue_pretrain/lr_decay.sh
```

For each non-empty subset $\mathcal{X} \subseteq \{\mathcal{A}, \mathcal{S}, \mathcal{F}\}$ you want, re-point `data_local` and `run_name` and re-run. That gives the 7 Phase-1 + 7 Phase-2 = 14 joint-training checkpoints in Table 2 of the paper.

`utils/convert_to_hf.sh` converts a Composer `.pt` checkpoint into HF Transformers FlexBERT format; `utils/upload_to_hub.sh` then pushes it.

### 4. Parameter-space merging (§4)

The 7 merged variants in Table 3 come from `model_merging/merge_configs/`:

- `merge_all.yaml` — uniform Soup of {A, F, S}
- `merge_ta_lambda{05,10}.yaml` — Task Arithmetic, $\lambda \in \{0.5, 1.0\}$
- `merge_ties_d{05,07}.yaml` — TIES, drop ratio $d \in \{0.5, 0.7\}$
- `merge_dare_d{05,07}.yaml` — DARE-TIES, $d \in \{0.5, 0.7\}$
- `merge_drop_{A,F,S}.yaml` — drop-one Soups for the §5.2 ablation (radar plot)

Drop your HF repo IDs into each YAML (the placeholder is `xxx/CMB_A`, etc.) and:

```bash
cd model_merging
bash merge.sh
```

### 5. Downstream evaluation (§4.3, §5)

```bash
cd eval
# n=3 seeds, all 9 tasks
bash run_multitask_pipeline.sh
# or, for the §5.1 paired-seed synthetic ablation (n=10):
python multitask_finetuning_10seeds.py --config_file config_updated.json --seeds 42 123 456 7 13 21 34 55 89 144
python benchmark_evaluation_updated.py --config_file config_updated.json --seeds 42 123 456 7 13 21 34 55 89 144
```

Edit `eval/config_updated.json` to point `base_model_path` at the HF repo (or local dir) of the checkpoint you want to fine-tune. The shared fine-tuning recipe (LR 4e-5, eff. batch 64, up to 10 epochs with early stopping on val F1, BF16 fused AdamW) follows §4.3 of the paper and is in `defaults.training`.

---

## Datasets used in evaluation

All are loaded via `datasets.load_dataset`; ClimRetrieve is local CSV.

| Task | HF dataset |
|---|---|
| Climate Detection | `climatebert/climate_detection` |
| Climate Specificity | `climatebert/climate_specificity` |
| Climate Sentiment | `climatebert/climate_sentiment` |
| Commitments & Actions | `climatebert/climate_commitments_actions` |
| Net Zero & Reduction | `climatebert/netzero_reduction_data` |
| TCFD Recommendations | `climatebert/tcfd_recommendations` |
| Environmental Claims | `climatebert/environmental_claims` |
| WFB Nature | `ESGBERT/WaterForestBiodiversityNature_2200` |
| WXImpactBench | `Michaelyya/wximpactbench-1386` |
| ClimRetrieve | Local CSV — point `local_data_path` at the ClimRetrieve report-level file |

---

## Environment

Tested on 4× A100 (Phase 1 takes ~7 days on the academic corpus, less on the others; Phase 2 is ~1 day). Key versions are pinned in `continue_pretrain/requirements.txt` — most importantly `mosaicml>=0.22,<0.23`, `mosaicml-streaming==0.7.6`, `transformers==4.40.2`, `torch==2.3.0`, `flash-attn==2.6.3`.

The merge step needs `mergekit` (installed inline by `merge.sh`). The synthetic-data step needs a vLLM server with `Qwen/Qwen3-30B-A3B-Instruct-2507`.

The repo is set up for SLURM (you'll see `#SBATCH` directives and a `module load dev2025a cuda/12.x` at the top of each launcher). If you're on a different cluster, swap those lines and the `/scratch/xxx`, `/home/xxx`, `--account=xxx` placeholders for whatever your site uses.

---

## Citation

Anonymized for review.
