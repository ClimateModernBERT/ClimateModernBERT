/**
 * Translation layer between historical Hugging Face experiment names and the
 * paper's corpus notation.
 *
 * Hugging Face repository names are *historical experiment identifiers*: they
 * record which internal data shard a run consumed at the time it was launched.
 * The manuscript defines only three corpora — 𝒜 (academic), ℱ (climate web),
 * 𝒮 (synthetic) — so the site renders paper notation and shows the raw repo id
 * underneath. This file is the single place that mapping lives.
 *
 * See docs/model-naming.md for the prose version.
 */

export type CorpusKey = "A" | "F" | "S";
export type Stage = "phase1" | "phase2";

export interface LegacyToken {
  token: string;
  maps: CorpusKey | "stage" | "none";
  meaning: string;
}

export const legacyTokens: LegacyToken[] = [
  { token: "MARK",             maps: "A",     meaning: "Academic-source data component. One of two internal academic shards." },
  { token: "ZYDA",             maps: "A",     meaning: "A second academic-source data component. Not a fourth corpus — 𝒜 in the paper covers both." },
  { token: "SYN",              maps: "S",     meaning: "Synthetic climate data." },
  { token: "WX_SYN",           maps: "S",     meaning: "Synthetic climate data under an earlier naming stage." },
  { token: "FWEdu_V2",         maps: "F",     meaning: "FineWeb-Edu-derived web corpus, before the FastText filtering stage." },
  { token: "FWEdu_V2_FastTxt", maps: "F",     meaning: "FineWeb-Edu climate corpus after FastText climate filtering. This is the paper's ℱ. FastText is a filter, not a corpus." },
  { token: "MICH",             maps: "none",  meaning: "Early exploratory data shard; predates the corpus definitions in the manuscript." },
  { token: "QWEN3_30B_A3B",    maps: "none",  meaning: "Synthetic-generation experiment using Qwen3-30B-A3B." },
  { token: "QWEN35_122B",      maps: "none",  meaning: "Synthetic-generation experiment using Qwen3.5-122B-A10B — the generator described in §3.1." },
  { token: "CX",               maps: "stage", meaning: "Phase 1 · context-extension continued pretraining." },
  { token: "CX_LRD",           maps: "stage", meaning: "Phase 1 followed by Phase 2 · learning-rate-decay specialization." },
];

export const stageLabel: Record<Stage, string> = {
  phase1: "Phase 1 · Continued Pretraining",
  phase2: "Phase 2 · LRD Specialization",
};

export const stageShort: Record<Stage, string> = {
  phase1: "Phase 1",
  phase2: "Phase 2",
};

export const stageSuffix: Record<Stage, string> = {
  phase1: "CX",
  phase2: "CX_LRD",
};

export const corpusName: Record<CorpusKey, string> = {
  A: "Academic",
  F: "Climate Web",
  S: "Synthetic",
};

export const corpusSymbol: Record<CorpusKey, string> = { A: "𝒜", F: "ℱ", S: "𝒮" };

/** Paper notation for a corpus set, e.g. ["A","S"] → "{𝒜, 𝒮}". */
export function paperNotation(corpora: readonly CorpusKey[]): string {
  const order: CorpusKey[] = ["A", "S", "F"];
  const sorted = order.filter((c) => corpora.includes(c));
  return `{${sorted.map((c) => corpusSymbol[c]).join(", ")}}`;
}

/** Human-readable corpus phrase, e.g. "Academic + Synthetic". */
export function corpusPhrase(corpora: readonly CorpusKey[]): string {
  const order: CorpusKey[] = ["A", "S", "F"];
  return order.filter((c) => corpora.includes(c)).map((c) => corpusName[c]).join(" + ");
}

/**
 * Best-effort parse of a historical repo name into stage + corpora.
 *
 * This exists so a newly uploaded checkpoint can be classified without hand-
 * editing the manifest, and so the manifest can be spot-checked against it.
 * It is deliberately conservative: entries in models.ts always win, because a
 * name alone is not sufficient evidence (see the MARK/ZYDA rule).
 */
export function parseLegacyName(repo: string): {
  stage: Stage | null;
  corpora: CorpusKey[];
  generator: string | null;
} {
  const name = repo.includes("/") ? repo.split("/")[1] : repo;
  // Underscore is a word character, so \\b cannot delimit these tokens. Split instead.
  const tokens = name.split("_");
  const has = (t: string) => tokens.some((x) => x.toUpperCase() === t);

  const stage: Stage | null = /_CX_LRD$/.test(name)
    ? "phase2"
    : /_CX$/.test(name)
      ? "phase1"
      : null;

  const corpora = new Set<CorpusKey>();
  if (has("MARK") || has("ZYDA")) corpora.add("A");
  if (tokens.some((t) => /^FWEdu$/i.test(t))) corpora.add("F");
  if (has("SYN")) corpora.add("S");

  let generator: string | null = null;
  if (/QWEN35_122B/.test(name)) generator = "Qwen3.5-122B-A10B";
  else if (/QWEN3_30B_A3B/.test(name)) generator = "Qwen3-30B-A3B";

  const order: CorpusKey[] = ["A", "S", "F"];
  return { stage, corpora: order.filter((c) => corpora.has(c)), generator };
}
