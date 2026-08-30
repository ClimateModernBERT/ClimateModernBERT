/**
 * Evaluation results, transcribed verbatim from the manuscript.
 *
 * Table 2  → phase1Rows / phase2Rows (+ baselines)
 * Table 3  → syntheticAblation
 * Table 4  → mergeRows
 * Table 7  → baselines[2] (Appendix G)
 * Table 9  → tasks
 * Table 10 → cxMergeRows (Appendix F)
 *
 * Scores are F1 (%), mean ± standard deviation. Binary tasks and ClimRetrieve
 * report positive-class F1; multi-class and multi-label tasks report macro-F1.
 * DO NOT edit a number here without checking it against the PDF.
 */

export const taskOrder = [
  "Retr.", "Comm.", "Det.", "Spec.", "Sent.", "NetZ.", "TCFD", "WFB", "WXI",
] as const;

export type TaskKey = (typeof taskOrder)[number];

/** [mean, stdev] per task, in taskOrder, followed by the average. */
export interface ScoreRow {
  label: string;
  /** Paper notation where one applies, e.g. "{A, S}". */
  variant?: string;
  scores: [number, number][];
  avg: [number, number];
  note?: string;
}

export const baselines: ScoreRow[] = [
  {
    label: "ModernBERT-base (θbase, stable-phase)",
    note: "The initialization used for every adapted variant.",
    scores: [[86.7,1.9],[65.9,1.3],[94.5,0.3],[67.3,0.7],[71.8,4.1],[98.7,0.6],[60.1,0.7],[96.0,0.4],[20.4,5.0]],
    avg: [73.5,1.7],
  },
  {
    label: "ClimateBERT",
    note: "RoBERTa-based climate-adapted encoder, fine-tuned under the identical protocol.",
    scores: [[82.4,2.6],[70.9,0.0],[97.5,0.0],[67.8,0.0],[76.6,0.0],[99.1,0.0],[51.4,0.1],[92.5,0.0],[3.3,4.6]],
    avg: [72.1,0.8],
  },
  {
    label: "ModernBERT-base (public post-LRD release)",
    note: "Reported in Appendix G for completeness; not the primary baseline.",
    scores: [[83.1,1.5],[60.6,1.5],[94.4,0.4],[67.4,2.0],[74.9,2.5],[98.9,0.3],[57.9,0.3],[96.8,0.1],[20.2,0.7]],
    avg: [72.7,1.0],
  },
];

/** Table 2, upper block — CX + LRD specialization (Phase 1 + Phase 2). */
export const phase2Rows: ScoreRow[] = [
  { label: "𝒜",         variant: "{A}",       scores: [[81.1,4.1],[68.7,2.1],[93.6,0.2],[67.3,1.1],[77.1,0.7],[98.0,1.2],[61.3,0.5],[96.9,0.3],[25.2,3.2]], avg: [74.4,2.4] },
  { label: "𝒮",         variant: "{S}",       scores: [[84.5,2.1],[66.4,3.4],[93.8,0.2],[69.4,0.9],[74.8,2.7],[99.0,0.1],[61.1,0.2],[96.5,0.3],[25.0,2.4]], avg: [74.5,1.4] },
  { label: "ℱ",         variant: "{F}",       scores: [[85.0,2.5],[67.4,1.6],[95.8,0.0],[68.7,1.9],[77.5,0.4],[98.7,0.0],[60.3,1.7],[96.8,0.2],[20.3,4.4]], avg: [74.5,1.4] },
  { label: "𝒜 + 𝒮",     variant: "{A, S}",    scores: [[83.2,4.1],[66.1,1.0],[93.2,0.8],[69.5,1.0],[78.1,0.5],[98.9,0.2],[61.3,1.0],[96.2,0.4],[24.3,4.9]], avg: [74.5,1.5] },
  { label: "𝒜 + ℱ",     variant: "{A, F}",    scores: [[83.5,4.4],[69.5,0.6],[95.5,0.6],[70.2,0.7],[76.0,0.4],[99.1,0.1],[63.0,2.5],[97.0,0.5],[18.5,5.0]], avg: [74.7,1.6] },
  { label: "𝒮 + ℱ",     variant: "{S, F}",    scores: [[85.3,2.2],[62.2,2.3],[95.7,0.2],[71.0,0.8],[75.5,0.6],[97.9,0.3],[59.4,2.2],[96.9,0.5],[23.4,3.7]], avg: [74.1,1.4] },
  { label: "𝒜 + 𝒮 + ℱ", variant: "{A, S, F}", scores: [[86.8,1.4],[60.0,2.3],[95.7,0.0],[71.8,1.4],[76.0,1.2],[98.6,0.4],[62.8,2.2],[97.5,0.2],[23.9,5.9]], avg: [74.8,1.7] },
];

/** Table 2, lower block — context-extension-only ablation (Phase 1). */
export const phase1Rows: ScoreRow[] = [
  { label: "𝒜",         variant: "{A}",       scores: [[85.5,1.0],[71.8,1.1],[93.2,0.3],[68.9,1.6],[77.4,1.0],[98.5,0.8],[59.0,0.9],[97.0,0.2],[26.1,3.2]], avg: [75.3,1.1] },
  { label: "𝒮",         variant: "{S}",       scores: [[86.0,1.8],[65.2,1.0],[93.5,0.8],[68.0,0.6],[71.8,4.1],[98.7,0.4],[60.0,0.7],[97.2,0.2],[20.0,2.1]], avg: [73.4,1.3] },
  { label: "ℱ",         variant: "{F}",       scores: [[85.4,1.0],[64.8,3.4],[94.5,0.3],[68.0,1.5],[76.3,0.5],[99.2,0.1],[59.1,1.0],[97.2,0.1],[23.4,5.1]], avg: [74.1,1.4] },
  { label: "𝒜 + 𝒮",     variant: "{A, S}",    scores: [[86.1,1.9],[67.9,1.4],[94.0,0.6],[67.6,0.9],[77.0,2.0],[98.2,0.9],[61.4,0.5],[96.6,0.5],[24.8,3.4]], avg: [74.8,1.3] },
  { label: "𝒜 + ℱ",     variant: "{A, F}",    scores: [[86.0,0.8],[65.9,0.7],[93.8,0.4],[67.2,1.1],[76.8,1.0],[99.1,0.2],[59.7,1.2],[96.5,0.3],[24.1,3.2]], avg: [74.3,1.0] },
  { label: "𝒮 + ℱ",     variant: "{S, F}",    scores: [[85.4,1.0],[64.8,3.4],[94.5,0.3],[68.0,1.5],[76.3,0.5],[99.2,0.1],[59.1,1.0],[97.2,0.1],[23.4,5.1]], avg: [74.2,1.4] },
  { label: "𝒜 + 𝒮 + ℱ", variant: "{A, S, F}", scores: [[86.5,1.0],[67.4,1.3],[95.2,0.1],[70.3,0.2],[75.2,0.6],[99.1,0.1],[59.2,1.1],[96.5,0.9],[18.5,5.3]], avg: [74.1,1.2] },
];

/** Table 4 — parameter-space merges of the three single-source Phase-2 checkpoints. */
export const mergeRows: ScoreRow[] = [
  { label: "θSoup",       note: "Simple weight averaging",     scores: [[83.6,1.6],[66.5,3.7],[95.4,0.6],[70.9,2.1],[76.7,1.4],[98.9,0.0],[64.1,2.2],[97.3,0.1],[33.7,5.3]], avg: [76.3,1.9] },
  { label: "θTA(1.0)",    note: "Task Arithmetic, λ = 1.0",    scores: [[85.0,0.4],[71.2,1.2],[94.9,0.5],[68.5,0.7],[77.0,3.1],[99.2,0.1],[61.3,0.7],[97.1,0.2],[27.2,5.1]], avg: [75.7,1.3] },
  { label: "θTIES(0.7)",  note: "TIES, drop ratio d = 0.7",    scores: [[85.1,2.8],[71.9,2.7],[94.9,0.1],[70.7,0.1],[75.6,1.6],[99.0,0.2],[59.7,1.1],[96.7,1.3],[26.9,3.8]], avg: [75.6,1.5] },
  { label: "θTIES(0.5)",  note: "TIES, drop ratio d = 0.5",    scores: [[86.1,1.4],[67.6,2.5],[93.6,0.6],[68.9,0.9],[77.1,0.7],[99.0,0.1],[61.4,0.6],[96.5,1.2],[28.5,1.7]], avg: [75.4,1.1] },
  { label: "θDARE(0.5)",  note: "DARE-TIES, drop ratio d = 0.5", scores: [[86.3,0.9],[67.2,0.4],[94.8,0.3],[69.2,0.2],[74.3,0.1],[98.9,0.2],[57.8,1.3],[97.1,0.2],[26.9,2.3]], avg: [74.7,0.7] },
  { label: "θDARE(0.7)",  note: "DARE-TIES, drop ratio d = 0.7", scores: [[85.5,0.9],[63.3,1.1],[94.7,0.1],[67.6,0.3],[74.1,2.0],[98.9,0.2],[61.6,1.0],[97.0,0.3],[26.2,6.2]], avg: [74.3,1.3] },
  { label: "θTA(0.5)",    note: "Task Arithmetic, λ = 0.5",    scores: [[80.9,5.7],[67.5,1.8],[94.7,0.4],[67.9,0.2],[71.0,0.2],[99.1,0.0],[60.3,2.4],[95.8,0.5],[24.8,6.0]], avg: [73.6,1.9] },
];

/** Table 10 (Appendix F) — merging with Phase-1 components, plus θNorm. */
export const cxMergeRows: ScoreRow[] = [
  { label: "θNorm",      note: "Norm-balanced linear merge · Phase-2 components", scores: [[84.3,1.8],[67.3,1.7],[94.1,1.1],[67.5,1.0],[78.0,1.5],[98.2,0.1],[61.4,0.7],[96.4,0.3],[23.8,3.4]], avg: [74.6,1.3] },
  { label: "θNorm (CX)", note: "Norm-balanced linear merge · Phase-1 components", scores: [[86.5,0.4],[70.1,1.7],[94.4,1.1],[68.8,1.0],[77.8,1.5],[98.3,0.5],[61.2,0.7],[96.5,0.5],[26.8,2.5]], avg: [75.9,1.2] },
  { label: "θTA(0.5) (CX)", note: "Task Arithmetic λ = 0.5 · Phase-1 components", scores: [[86.4,0.8],[67.1,1.6],[94.7,1.1],[68.2,1.0],[78.6,3.1],[98.1,0.1],[61.0,1.0],[96.3,0.2],[20.9,7.1]], avg: [74.6,1.8] },
  { label: "θSoup (CX)", note: "Simple weight averaging · Phase-1 components",    scores: [[84.5,1.6],[66.5,2.6],[94.1,0.3],[69.6,0.7],[78.2,2.8],[98.3,0.1],[61.3,1.5],[96.5,0.2],[22.5,8.3]], avg: [74.2,1.5] },
];

/**
 * Table 3 — paired-seed synthetic-data ablation (n = 10). Δ = mean effect of adding 𝒮.
 * NOT currently rendered: the page summarises this in Key Finding 02 instead.
 */
export const syntheticAblation = {
  caption:
    "Effect of adding 𝒮 to 𝒜, measured over ten shared fine-tuning seeds. pt is the paired-t p-value, pw the Wilcoxon signed-rank p-value. Bold marks significance (α = 0.05) under both tests.",
  rows: [
    { task: "Retrieval",   cx: { d: +0.6, pt: ".31",   pw: ".26",  sig: false }, lrd: { d: +2.2, pt: ".66",  pw: ".77",  sig: false } },
    { task: "Commitments", cx: { d: -3.9, pt: "< .001",pw: ".002", sig: true  }, lrd: { d: -2.6, pt: ".016", pw: ".037", sig: true  } },
    { task: "Detection",   cx: { d: +0.8, pt: ".012",  pw: ".020", sig: true  }, lrd: { d: -0.4, pt: ".18",  pw: ".38",  sig: false } },
    { task: "Specificity", cx: { d: -1.3, pt: ".07",   pw: ".13",  sig: false }, lrd: { d: +2.2, pt: ".001", pw: ".002", sig: true  } },
    { task: "Sentiment",   cx: { d: -0.4, pt: ".43",   pw: ".56",  sig: false }, lrd: { d: +0.9, pt: ".007", pw: ".006", sig: true  } },
    { task: "NetZero",     cx: { d: -0.3, pt: ".51",   pw: ".63",  sig: false }, lrd: { d: +0.9, pt: ".07",  pw: ".08",  sig: false } },
    { task: "TCFD",        cx: { d: +2.4, pt: "< .001",pw: ".002", sig: true  }, lrd: { d: -0.0, pt: ".96",  pw: ".92",  sig: false } },
    { task: "WFB",         cx: { d: -0.4, pt: ".06",   pw: ".08",  sig: false }, lrd: { d: -0.8, pt: ".001", pw: ".004", sig: true  } },
    { task: "WXImpact",    cx: { d: -1.3, pt: ".25",   pw: ".16",  sig: false }, lrd: { d: -0.9, pt: ".49",  pw: ".63",  sig: false } },
  ],
};

/** Figure 2 — drop-one Soup ablation. The manuscript reports deltas, not per-task tables. */
export const dropOne = {
  caption:
    "Average F1 change of a two-source Soup relative to the full Soup({𝒜, 𝒮, ℱ}).",
  rows: [
    { key: "A" as const, dropped: "Academic",    kept: "Soup({𝒮, ℱ})", delta: -4.0, note: "Largest degradation across nearly all benchmarks." },
    { key: "S" as const, dropped: "Synthetic",   kept: "Soup({𝒜, ℱ})", delta: -1.5, note: "Substantially smaller; improves some individual benchmarks." },
    { key: "F" as const, dropped: "Climate Web", kept: "Soup({𝒜, 𝒮})", delta: -1.5, note: "Substantially smaller; improves some individual benchmarks." },
  ],
  footnote:
    "The manuscript reports −4.0 for dropping 𝒜 and a common −1.5 figure for excluding either 𝒮 or ℱ; per-task values are shown in Figure 2 of the PDF.",
};

/** Table 9 — the nine downstream benchmarks. */
export const tasks = [
  { key: "Det.",  name: "Climate Detection",     type: "Binary",      classes: 2, train: 1170, test: 400, dataset: "climatebert/climate_detection",
    blurb: "Is a passage climate-related? Requires separating domain-relevant content from superficially similar but topically unrelated text." },
  { key: "Spec.", name: "Climate Specificity",   type: "Binary",      classes: 2, train: 900,  test: 320, dataset: "climatebert/climate_specificity",
    blurb: "Specific vs. vague: does the statement contain concrete commitments or measurable targets? Central to greenwashing detection." },
  { key: "Comm.", name: "Commitments & Actions", type: "Binary",      classes: 2, train: 900,  test: 320, dataset: "climatebert/climate_commitments_actions",
    blurb: "Forward-looking pledge vs. realized action — the gap between corporate rhetoric and implemented measures." },
  { key: "Sent.", name: "Climate Sentiment",     type: "Multi-class", classes: 3, train: 900,  test: 320, dataset: "climatebert/climate_sentiment",
    blurb: "Risk / neutral / opportunity: evaluative stance toward climate change in corporate and policy discourse." },
  { key: "NetZ.", name: "Net Zero & Reduction",  type: "Multi-class", classes: 3, train: 2753, test: 344, dataset: "climatebert/netzero_reduction_data",
    blurb: "Net-zero target, reduction target, or no target, from expert-annotated corporate reports and national pledges." },
  { key: "TCFD",  name: "TCFD Recommendations",  type: "Multi-class", classes: 4, train: 1170, test: 400, dataset: "climatebert/tcfd_recommendations",
    blurb: "Governance / strategy / risk / metrics-and-targets: which TCFD disclosure pillar a passage belongs to." },
  { key: "WFB",   name: "WFB Nature",            type: "Multi-label", classes: 4, train: 1760, test: 220, dataset: "ESGBERT/WaterForestBiodiversityNature_2200",
    blurb: "TNFD-aligned nature dimensions — water, forest, biodiversity, nature — that may co-occur in one passage." },
  { key: "WXI",   name: "WXImpactBench",         type: "Multi-label", classes: 6, train: 970,  test: 208, dataset: "Michaelyya/wximpactbench-1386",
    blurb: "Six societal-impact categories of disruptive weather events, annotated over historical newspaper articles." },
  { key: "Retr.", name: "ClimRetrieve",          type: "Retrieval",   classes: 2, train: 6800, test: 850, dataset: "Local CSV (report-level split)",
    blurb: "Passage relevance for 16 climate questions over 30 sustainability reports, binarized at relevance ≥ 1." },
] as const;

/**
 * §4.3 — the shared downstream fine-tuning recipe.
 * NOT currently rendered; kept as the transcription of record.
 */
export const finetuning = [
  ["Learning rate", "4 × 10⁻⁵"],
  ["Effective batch size", "64 (32 × 2 grad-accum steps)"],
  ["Weight decay", "0.01"],
  ["Epochs", "up to 10, early stopping on validation F1"],
  ["Precision", "BF16, fused AdamW"],
  ["Seeds", "3 (10 for the paired-seed synthetic ablation)"],
] as const;

/** Headline comparison used by the results chart. Labels stay short. */
export const headline = [
  { label: "ClimateBERT",            sub: "prior climate encoder",        value: 72.1, kind: "baseline" as const },
  { label: "ModernBERT-base",        sub: "θbase · initialization",       value: 73.5, kind: "baseline" as const },
  { label: "Joint 𝒜+𝒮+ℱ",            sub: "Phase 2 · full corpus union",  value: 74.8, kind: "joint" as const },
  { label: "Best single source 𝒜",   sub: "Phase 1 · academic only",      value: 75.3, kind: "joint" as const },
  { label: "θSoup",                  sub: "weight-averaged merge",        value: 76.3, kind: "merge" as const },
];
