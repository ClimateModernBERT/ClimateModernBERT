/**
 * The canonical GitHub location. Every other GitHub URL in this file is derived
 * from it, so moving the repository is a one-line change here.
 */
const GITHUB = "https://github.com/ClimateModernBERT/ClimateModernBERT";

/**
 * Project-level facts.
 *
 * SOURCE OF TRUTH: paper/climate-modernbert.pdf (manuscript, under review).
 * Every number here is transcribed from that PDF. If the paper changes, edit
 * this file — no component hard-codes these values.
 */

export const project = {
  name: "ClimateModernBERT",
  paperTitle:
    "Climate-ModernBERT: Revisiting Corpus Composition for Domain-Adaptive Continued Pretraining",
  tagline: "Revisiting Corpus Composition for Domain-Adaptive Continued Pretraining",
  /** The manuscript is unpublished / under review. Never claim a venue. */
  status: "Preprint manuscript · under review",
  blurb:
    "A family of climate-domain ModernBERT encoders. We continue-pretrain ModernBERT-Base on academic climate text, climate-filtered web data, and synthetic climate text, then ask which corpus composition — and which integration mechanism — actually helps.",
  abstract:
    "Natural Language Processing (NLP) in the climate domain requires models to process heterogeneous text sources, including scientific literature, policy disclosures, and synthetic reports. However, how to effectively combine diverse domain corpora during continued pretraining (CPT) remains underexplored. We introduce Climate-ModernBERT, a family of climate-adapted encoder models obtained through continued pretraining of ModernBERT-Base on three climate corpora: academic climate text, climate-filtered web data, and synthetic climate documents. We systematically compare joint continued pretraining on corpus mixtures with parameter-space merging of independently specialized checkpoints. Across nine climate NLP benchmarks, our best model achieves 76.3 average F1, improving significantly over a climate-adapted ModernBERT baseline by 2.8 points. Within the climate NLP setting, the results show that academic climate corpora provide the strongest adaptation signal among the evaluated sources, while parameter-space merging improves over joint multi-source training and better preserves complementary information from heterogeneous climate corpora. We release all Climate-ModernBERT variants and training checkpoints to support future research in climate NLP and domain-adaptive pretraining.",

  links: {
    github: GITHUB,
    /** The organization the paper's checkpoints are published under. */
    hfOrgPage: "https://huggingface.co/CMB-ClimateModernBERT",
    /** Curated "final models from the paper" collection under the legacy account. */
    hfCollection: "https://huggingface.co/collections/sraj/climatemodernbert",
    /** Full experimental collection, including legacy runs. */
    hfCollectionAll: "https://huggingface.co/collections/sraj/cmb-all",
    hfOrg: "https://huggingface.co/sraj",
    paperPdf: "climate-modernbert.pdf",
  },

  /** Headline counts, all from the manuscript (§1, §4.2). */
  scale: [
    { value: "21", label: "adapted variants", note: "7 Phase-1 · 7 Phase-2 · 7 merged" },
    { value: "9", label: "climate NLP benchmarks", note: "classification, multi-label, retrieval" },
    { value: "810+", label: "fine-tuning runs", note: "n=3 seeds, n=10 for the synthetic ablation" },
    { value: "6.42B", label: "pretraining tokens", note: "across three corpora" },
  ],

  /**
   * ModernBERT-Base architecture, Appendix E / Table 8.
   * NOT currently rendered; kept as the transcription of record.
   */
  architecture: [
    ["Parameters", "150M"],
    ["Layers", "22"],
    ["Hidden size", "768"],
    ["Intermediate size", "1,152"],
    ["Attention heads", "12 (head dim 64)"],
    ["Vocabulary", "50,368"],
    ["Context length", "8,192 tokens"],
    ["Position encoding", "RoPE (base 160K)"],
    ["Attention", "Alternating local (window 128) / global every 3 layers"],
  ] as const,

  /**
   * Continued-pretraining recipe, Appendix E / Table 8.
   * NOT currently rendered; kept as the transcription of record.
   */
  training: {
    phase1: [
      ["Epochs", "3"],
      ["Learning rate", "3 × 10⁻⁴ (constant)"],
      ["Warm-up", "0 epochs"],
      ["Global batch size", "576"],
      ["Sequence length", "8,192"],
      ["MLM masking rate", "30%"],
      ["Optimizer", "StableAdamW (β = 0.9/0.98, ε = 10⁻⁶, wd = 10⁻⁵)"],
      ["Precision", "BF16 (AMP)"],
    ],
    phase2: [
      ["Epochs", "3"],
      ["LR schedule", "1 − √t decay"],
      ["Initial LR", "3 × 10⁻⁴"],
      ["Final LR factor αf", "0.001"],
      ["Data", "variant-specific subset"],
      ["Other settings", "same as Phase 1"],
    ],
    compute: "4× NVIDIA A100 · MosaicML Composer · BF16 · gradient checkpointing",
  },

  /** §3.1 and Table 1 / Table 5 / Table 6. */
  corpora: [
    {
      key: "A",
      symbol: "A",
      name: "Academic",
      accent: "a",
      format: "XML / CSV / PDF",
      docs: "~5M",
      tokens: "~1.28B",
      rawTokens: "~4.65B before deduplication",
      summary:
        "Four streams of in-domain text: peer-reviewed journal articles (climate/earth science, energy and resource economics, climate-relevant finance), the ClimateNews archive 2000–2022, climate-focused arXiv preprints, and a small set of climate handbooks.",
      detail: [
        "Journal articles parsed from structured XML; section labels and captions retained, reference lists and boilerplate removed.",
        "arXiv preprints restricted to physics.ao-ph, econ.GN, q-fin.RM and climate-tagged cs, converted with a layout-aware extractor.",
        "Boilerplate stripping, English-only language ID, and a 200-token minimum-length filter before deduplication.",
      ],
      redistribution:
        "Not redistributable. Peer-reviewed articles are accessed under institutional publisher licenses; news shards and handbooks were collected for non-commercial research use. We release model checkpoints and processing pipelines, not the raw text.",
    },
    {
      key: "F",
      symbol: "F",
      name: "Climate Web",
      accent: "f",
      format: "Parquet",
      docs: "2.59M",
      tokens: "~5B",
      rawTokens: "derived from FineWeb-Edu",
      summary:
        "Climate-relevant documents extracted from FineWeb-Edu through a two-stage filtering pipeline: a high-recall keyword filter built from 166 climate-related terms, followed by deduplication.",
      detail: [
        "Keyword list drawn from the climate concept in the GEMET thesaurus and Sautner et al. (2023).",
        "A FastText climate classifier forms the second filtering stage — an implementation detail of ℱ, not a separate corpus.",
        "Deduplicated with the NeMo Curator pipeline.",
      ],
      redistribution:
        "Derived from the public FineWeb-Edu dataset; the filtering pipeline is released in this repository.",
    },
    {
      key: "S",
      symbol: "S",
      name: "Synthetic",
      accent: "s",
      format: "JSONL",
      docs: "~20K",
      tokens: "~0.14B",
      rawTokens: "6,785 seed documents × 3 styles",
      summary:
        "LLM-generated climate text conditioned on seed excerpts drawn from six in-domain sources — five academic journal subsets of 𝒜 plus the ClimateNews archive — to improve coverage of underrepresented climate subdomains.",
      detail: [
        "Each seed contributes its first 800 characters of cleaned body text.",
        "Three climate-communication styles: public awareness, industry perspective, environmental journalism.",
        "Generated once per (seed, style) pair with Qwen3.5-122B-A10B at temperature 0.6, top-p 0.95, 1,024 new tokens.",
      ],
      redistribution:
        "Model-generated text. Generation scripts and prompt templates are in src/synthetic/.",
    },
  ],

  /** §5, transcribed. Do not overstate beyond the manuscript. */
  findings: [
    {
      n: "01",
      title: "Academic text is the strongest signal",
      body:
        "Academic-only adaptation reaches the highest average F1 of any jointly trained configuration. Gains concentrate on register-sensitive tasks — Sentiment +6.3, WXImpactBench +7.5, Commitments +5.9.",
      stat: "75.3",
      statLabel: "avg F1 · academic only",
    },
    {
      n: "02",
      title: "More sources are not better sources",
      body:
        "Adding web and synthetic text to the academic corpus lowers average F1 from 75.3 to 74.1. Synthetic text is the clearest case: it significantly helps TCFD and Specificity while consistently degrading Commitments & Actions.",
      stat: "−1.2",
      statLabel: "avg F1 · 𝒜 → 𝒜+𝒮+ℱ",
    },
    {
      n: "03",
      title: "Merge afterwards, don't mix upfront",
      body:
        "Averaging three independently adapted checkpoints beats joint training on the very same corpus union. Their task vectors are near-orthogonal (pairwise cosine 0.06–0.18), so averaging keeps what each source learned.",
      stat: "76.3",
      statLabel: "avg F1 · θSoup",
    },
  ],

  /**
   * Repository map. NOT currently rendered — the README covers reproduction.
   */
  repoMap: [
    {
      dir: "src/data/ + continue_pretrain/data_pipeline/",
      title: "Data preparation",
      body:
        "Academic XML/PDF extraction, suffix-array deduplication, decontamination against all nine eval sets, FineWeb-Edu streaming filter, and the NeMo-Curator clean → exact-dedup → fuzzy-dedup pipeline.",
      href: `${GITHUB}/tree/main/src/data`,
    },
    {
      dir: "src/synthetic/",
      title: "Synthetic generation",
      body:
        "Seed selection, the three style templates, and the vLLM driver that produces the 𝒮 corpus. Three SLURM jobs at seeds 42 / 43 / 44 for variance.",
      href: `${GITHUB}/tree/main/src/synthetic`,
    },
    {
      dir: "continue_pretrain/",
      title: "Continued pretraining",
      body:
        "Composer / FlexBERT configs for Phase 1 (context extension) and Phase 2 (LRD specialization), plus the SLURM launchers. Re-point data_local and run_name for each corpus subset.",
      href: `${GITHUB}/tree/main/continue_pretrain`,
    },
    {
      dir: "model_merging/",
      title: "Parameter-space merging",
      body:
        "Ten mergekit configurations covering Soup, Task Arithmetic, TIES, DARE-TIES, and the drop-one ablations, with a driver that runs mergekit and pushes to the Hub.",
      href: `${GITHUB}/tree/main/model_merging`,
    },
    {
      dir: "eval/",
      title: "Evaluation",
      body:
        "Multitask fine-tuning across the nine benchmarks at n=3 seeds, the n=10 paired-seed variant used for the synthetic ablation, and the scoring harness.",
      href: `${GITHUB}/tree/main/eval`,
    },
    {
      dir: "utils/",
      title: "Conversion & release",
      body:
        "JSONL/HF → MosaicML Streaming (MDS) shard packing, Composer checkpoint → HF Transformers conversion, and Hub upload helpers.",
      href: `${GITHUB}/tree/main/utils`,
    },
  ],
} as const;

export type Corpus = (typeof project.corpora)[number];
