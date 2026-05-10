"""
Polished n=10 + merged-model analysis for ClimateModernBERT.

Inputs (in this directory):
- ClimateModernBERT results - April 28 - seed 10.csv         (10 seeds, 4 models)
- ClimateModernBERT results - April 28 - Merged model.csv    (3 seeds, 7 merge variants)
- ClimateModernBERT_results_epoch_comparison.csv             (3 seeds, all model variants)
- ClimateModernBERT_results_epoch10.csv                      (3 seeds, baseline)

Outputs:
- summary_per_model_n10.csv          mean/std for the 4 n=10 models
- ablation_n10_stats.csv             SYN/LRD ablation w/ paired-t + Wilcoxon + Cohen's d
- merge_vs_components.csv            each merge variant vs (MARK+LRD, FWEdu+LRD, SYN+LRD)
- merge_vs_joint.csv                 each merge variant vs joint-trained baselines
- per_task_ranking.csv               every model ranked per task
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

ROOT = Path(__file__).resolve().parent

# Primary metric per task (matches previous analysis).
PRIMARY_METRIC = {
    "clim_retrieve": "f1",
    "climate_commitments_actions": "f1",
    "climate_detection": "f1",
    "climate_specificity": "f1",
    "climate_sentiment": "macro_f1",
    "environmental_claims": "f1",
    "netzero_reduction": "macro_f1",
    "tcfd_recommendations": "macro_f1",
    "water_forest_biodiversity_nature": "macro_f1",
    "wximpactbench": "macro_f1",
}

VALUE_RE = re.compile(
    r"(?P<metric>[\w@]+):\s*(?P<mean>-?\d+\.\d+)\s*±\s*(?P<std>-?\d+\.\d+)\s*\[(?P<seeds>[^\]]*)\]"
)


def parse_cell(cell: str) -> tuple[str, float, float, list[float]] | None:
    cell = (cell or "").strip().strip('"')
    if not cell:
        return None
    m = VALUE_RE.match(cell)
    if not m:
        return None
    metric = m.group("metric")
    mean_v = float(m.group("mean"))
    std_v = float(m.group("std"))
    seeds = [float(x) for x in m.group("seeds").split(",") if x.strip()]
    return metric, mean_v, std_v, seeds


def parse_results_csv(path: Path) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Return: model_name -> task -> metric -> [per-seed values]."""
    with path.open() as f:
        rows = list(csv.reader(f))

    # Find a header row containing model names. Some files start with a blank line.
    header = None
    start_idx = 0
    for i, row in enumerate(rows):
        if any(c.strip() for c in row) and not any(":" in c for c in row):
            header = [c.strip() for c in row]
            start_idx = i + 1
            break
    if header is None:
        raise ValueError(f"No header found in {path}")

    models: dict[str, dict[str, dict[str, list[float]]]] = {
        h: {} for h in header if h
    }
    current_task: dict[str, str] = {h: "" for h in header}

    for row in rows[start_idx:]:
        for col_idx, cell in enumerate(row):
            if col_idx >= len(header):
                continue
            model = header[col_idx]
            if not model:
                continue
            cell_s = (cell or "").strip()
            if not cell_s:
                continue
            # Task header line ends with ":"
            if cell_s.endswith(":") and "±" not in cell_s:
                current_task[model] = cell_s.rstrip(":").strip()
                models[model].setdefault(current_task[model], {})
                continue
            parsed = parse_cell(cell)
            if parsed is None:
                continue
            metric, _, _, seeds = parsed
            task = current_task.get(model, "")
            if not task:
                continue
            models[model].setdefault(task, {})[metric] = seeds
    return models


def primary_seeds(model_data: dict[str, dict[str, list[float]]], task: str) -> list[float] | None:
    metric = PRIMARY_METRIC.get(task)
    if metric is None:
        return None
    return model_data.get(task, {}).get(metric)


def cohens_d_paired(a: list[float], b: list[float]) -> float:
    """Sign-aligned with delta = mean(b) - mean(a). Positive => B > A."""
    diffs = [y - x for x, y in zip(a, b)]
    if len(diffs) < 2:
        return float("nan")
    sd = stdev(diffs)
    if sd == 0:
        return float("nan")
    return mean(diffs) / sd


def paired_tests(a: list[float], b: list[float]) -> dict[str, float]:
    diffs = [y - x for x, y in zip(a, b)]  # B - A
    out: dict[str, float] = {
        "n": len(a),
        "mean_a": mean(a),
        "mean_b": mean(b),
        "delta": mean(b) - mean(a),
        "cohens_d": cohens_d_paired(a, b),
    }
    # All differences exactly zero (deterministic): tests undefined.
    if all(d == 0 for d in diffs):
        out["paired_t_p"] = float("nan")
        out["wilcoxon_p"] = float("nan")
        out["direction"] = "identical"
        return out
    # Paired t
    if len(a) >= 2 and stdev(diffs) > 0:
        t = stats.ttest_rel(b, a)  # tests B - A
        out["paired_t_p"] = float(t.pvalue)
    else:
        out["paired_t_p"] = float("nan")
    # Wilcoxon
    try:
        w = stats.wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
        out["wilcoxon_p"] = float(w.pvalue)
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    # Direction consistency: B vs A on the *primary* metric.
    ups = sum(1 for d in diffs if d > 0)
    downs = sum(1 for d in diffs if d < 0)
    if ups == len(diffs):
        out["direction"] = "all_B>A"
    elif downs == len(diffs):
        out["direction"] = "all_B<A"
    else:
        out["direction"] = "mixed"
    return out


# ---- Load all CSVs ----
seed10 = parse_results_csv(ROOT / "ClimateModernBERT results - April 28 - seed 10.csv")
merge = parse_results_csv(ROOT / "ClimateModernBERT results - April 28 - Merged model.csv")
old3 = parse_results_csv(ROOT / "ClimateModernBERT_results_epoch_comparison.csv")

# Map raw column names to friendly ones used in the older delta CSV.
SEED10_RENAME = {
    "CMB_MARK_CX": "MARK",
    "CMB_MARK_WX_SYN_CX": "MARK+SYN",
    "CMB_MARK_WX_SYN_CX_LRD": "MARK+SYN+LRD",
    "CMB_MARK_CX_LRD": "MARK+LRD",
}
seed10 = {SEED10_RENAME.get(k, k): v for k, v in seed10.items()}

OLD_RENAME = {
    "BASE-MODEL": "BASE",
    "CMB_MARK_WX_SYN_CX_LRD": "MARK+SYN+LRD_3seed",
    "CMB_MARK_CX_LRD": "MARK+LRD_3seed",
    "CMB_FWEdu_V2_CX_LRD": "FWEdu+LRD",
    "CMB_WX_SYN_CX_LRD": "SYN+LRD",
    "CMB_MARK_WX_SYN_CX": "MARK+SYN_3seed",
    "CMB_MARK_CX": "MARK_3seed",
    "CMB_WX_SYN_CX": "SYN",
    "CMB_FWEdu_V2_CX": "FWEdu",
    "CMB_FWEdu_V2_MARK_WX_SYN_CX": "FWEdu+MARK+SYN",
    "CMB_FWEdu_V2_MARK_WX_SYN_ZYDA_CX": "FWEdu+MARK+SYN+ZYDA",
    "CMB_MARK_WX_SYN_ZYDA_CX": "MARK+SYN+ZYDA",
    "CMB_WX_SYN_ZYDA_CX": "SYN+ZYDA",
    "CMB_FWEdu_V2_MARK_WX_SYN_CX_LRD": "FWEdu+MARK+SYN+LRD",
    "CMB_FWEdu_V2_MARK_WX_SYN_ZYDA_CX_LRD": "FWEdu+MARK+SYN+ZYDA+LRD",
}
old3 = {OLD_RENAME.get(k, k): v for k, v in old3.items()}

MERGE_RENAME = {
    "Merge_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD": "Merge_avg",
    "DARE_TIES_D05_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD": "DARE_TIES_D05",
    "DARE_TIES_D07_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD": "DARE_TIES_D07",
    "TA_Lambda05_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD": "TA_Lambda05",
    "TA_Lambda10_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD": "TA_Lambda10",
    "TIES_D05_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD": "TIES_D05",
    "TIES_D07_CMB_MARK_CX_LRD_CMB_FWEdu_V2_CX_LRD_CMB_WX_SYN_CX_LRD": "TIES_D07",
}
merge = {MERGE_RENAME.get(k, k): v for k, v in merge.items()}

TASKS = list(PRIMARY_METRIC.keys())

# environmental_claims has 0 variance across all seeds (deterministic finetune
# on a tiny dev set), so significance tests don't apply. We still report means
# in the per-task ranking but exclude from paired statistical tests.
STAT_TASKS = [t for t in TASKS if t != "environmental_claims"]


# ---- 1) Per-model summary for the 4 n=10 models ----
with (ROOT / "summary_per_model_n10.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model", "Task", "Metric", "N_seeds", "Mean", "Std"])
    for model, data in seed10.items():
        for task in TASKS:
            metric = PRIMARY_METRIC[task]
            seeds = data.get(task, {}).get(metric)
            if not seeds:
                continue
            w.writerow([
                model, task, metric, len(seeds),
                f"{mean(seeds):.4f}",
                f"{stdev(seeds):.4f}" if len(seeds) > 1 else "0.0000",
            ])


# ---- 2) SYN/LRD ablations at n=10 ----
ABLATIONS_N10 = [
    ("SYN_ablation",       "MARK",        "MARK+SYN"),
    ("SYN_ablation_LRD",   "MARK+LRD",    "MARK+SYN+LRD"),
    ("LRD_ablation",       "MARK",        "MARK+LRD"),
    ("LRD_ablation_SYN",   "MARK+SYN",    "MARK+SYN+LRD"),
]

with (ROOT / "ablation_n10_stats.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "Test", "Model_A", "Model_B", "Task", "Metric", "N",
        "Mean_A", "Mean_B", "Delta",
        "Paired_t_p", "Wilcoxon_p", "Cohens_d",
        "Direction", "Sig_0.05", "Sig_0.10",
    ])
    for test_name, A, B in ABLATIONS_N10:
        for task in STAT_TASKS:
            metric = PRIMARY_METRIC[task]
            a = seed10[A].get(task, {}).get(metric)
            b = seed10[B].get(task, {}).get(metric)
            if not a or not b or len(a) != len(b):
                continue
            r = paired_tests(a, b)
            w.writerow([
                test_name, A, B, task, metric, r["n"],
                f"{r['mean_a']:.4f}", f"{r['mean_b']:.4f}", f"{r['delta']:+.4f}",
                f"{r['paired_t_p']:.4f}" if not math.isnan(r['paired_t_p']) else "NaN",
                f"{r['wilcoxon_p']:.4f}" if not math.isnan(r['wilcoxon_p']) else "NaN",
                f"{r['cohens_d']:.2f}" if not math.isinf(r['cohens_d']) and not math.isnan(r['cohens_d']) else str(r['cohens_d']),
                r["direction"],
                "Yes" if not math.isnan(r["paired_t_p"]) and r["paired_t_p"] < 0.05 else "No",
                "Yes" if not math.isnan(r["paired_t_p"]) and r["paired_t_p"] < 0.10 else "No",
            ])


# ---- 3) Merge vs components (per-seed paired comparison; n=3) ----
COMPONENTS = [
    ("MARK+LRD",  "MARK+LRD"),   # use the n=10 MARK+LRD if available, else 3-seed
    ("FWEdu+LRD", "FWEdu+LRD"),
    ("SYN+LRD",   "SYN+LRD"),
]


def get_seeds(table: dict[str, dict[str, dict[str, list[float]]]],
              model: str, task: str) -> list[float] | None:
    metric = PRIMARY_METRIC[task]
    return table.get(model, {}).get(task, {}).get(metric)


def take_first_n(xs: list[float] | None, n: int) -> list[float] | None:
    if xs is None:
        return None
    return xs[:n]


with (ROOT / "merge_vs_components.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "Merge_method", "Component", "Task", "Metric", "N",
        "Mean_Component", "Mean_Merge", "Delta",
        "Paired_t_p", "Wilcoxon_p", "Cohens_d",
        "Direction",
    ])
    for method, mdata in merge.items():
        for comp_label, comp_model in COMPONENTS:
            for task in STAT_TASKS:
                metric = PRIMARY_METRIC[task]
                a = get_seeds(seed10, comp_model, task) or get_seeds(old3, comp_model, task)
                b = mdata.get(task, {}).get(metric)
                if not a or not b:
                    continue
                # Pair on first min(len) seeds
                n = min(len(a), len(b))
                if n < 2:
                    continue
                a_ = a[:n]
                b_ = b[:n]
                r = paired_tests(a_, b_)
                w.writerow([
                    method, comp_label, task, metric, n,
                    f"{r['mean_a']:.4f}", f"{r['mean_b']:.4f}", f"{r['delta']:+.4f}",
                    f"{r['paired_t_p']:.4f}" if not math.isnan(r['paired_t_p']) else "NaN",
                    f"{r['wilcoxon_p']:.4f}" if not math.isnan(r['wilcoxon_p']) else "NaN",
                    f"{r['cohens_d']:.2f}" if not math.isinf(r['cohens_d']) and not math.isnan(r['cohens_d']) else str(r['cohens_d']),
                    r["direction"],
                ])


# ---- 4) Merge vs joint training ----
JOINT_BASELINES = [
    "FWEdu+MARK+SYN+LRD",
    "FWEdu+MARK+SYN+ZYDA+LRD",
    "MARK+SYN+LRD",   # n=10
]

with (ROOT / "merge_vs_joint.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "Merge_method", "Joint_baseline", "Task", "Metric", "N",
        "Mean_Joint", "Mean_Merge", "Delta",
        "Paired_t_p", "Wilcoxon_p", "Cohens_d",
        "Direction",
    ])
    for method, mdata in merge.items():
        for joint in JOINT_BASELINES:
            for task in STAT_TASKS:
                metric = PRIMARY_METRIC[task]
                a = get_seeds(seed10, joint, task) or get_seeds(old3, joint, task)
                b = mdata.get(task, {}).get(metric)
                if not a or not b:
                    continue
                n = min(len(a), len(b))
                if n < 2:
                    continue
                a_ = a[:n]
                b_ = b[:n]
                r = paired_tests(a_, b_)
                w.writerow([
                    method, joint, task, metric, n,
                    f"{r['mean_a']:.4f}", f"{r['mean_b']:.4f}", f"{r['delta']:+.4f}",
                    f"{r['paired_t_p']:.4f}" if not math.isnan(r['paired_t_p']) else "NaN",
                    f"{r['wilcoxon_p']:.4f}" if not math.isnan(r['wilcoxon_p']) else "NaN",
                    f"{r['cohens_d']:.2f}" if not math.isinf(r['cohens_d']) and not math.isnan(r['cohens_d']) else str(r['cohens_d']),
                    r["direction"],
                ])


# ---- 5) Per-task ranking across all known models (mean of primary metric) ----
# Prefer the n=10 version for the four shared models; rename old 3-seed entries
# back to their canonical names (we'd already suffixed them with _3seed for
# safety in stat tests above, but for ranking the 10-seed runs are authoritative).
all_models: dict[str, dict[str, dict[str, list[float]]]] = {}
for k, v in seed10.items():
    all_models[k] = v
for k, v in old3.items():
    canonical = k.replace("_3seed", "")
    if canonical in all_models:
        continue  # n=10 wins
    all_models[canonical] = v
for k, v in merge.items():
    all_models[k] = v

# For tasks where we have a baseline, compute delta from BASE.
base = old3.get("BASE", {})


def base_score(task: str) -> float | None:
    metric = PRIMARY_METRIC[task]
    seeds = base.get(task, {}).get(metric)
    if not seeds:
        return None
    return mean(seeds)


with (ROOT / "per_task_ranking.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Task", "Metric", "Rank", "Model", "N_seeds", "Mean", "Std", "Delta_from_BASE"])
    for task in TASKS:
        metric = PRIMARY_METRIC[task]
        rows = []
        for model, data in all_models.items():
            seeds = data.get(task, {}).get(metric)
            if not seeds:
                continue
            rows.append((
                model, len(seeds), mean(seeds),
                stdev(seeds) if len(seeds) > 1 else 0.0,
            ))
        rows.sort(key=lambda x: x[2], reverse=True)
        b = base_score(task)
        for rank, (model, n, mu, sd) in enumerate(rows, 1):
            d = "" if b is None else f"{mu - b:+.4f}"
            w.writerow([task, metric, rank, model, n, f"{mu:.4f}", f"{sd:.4f}", d])


# ---- 6) Average delta from base, ranked across all models ----
deltas: dict[str, list[float]] = {}
for model, data in all_models.items():
    if model == "BASE":
        continue
    ds = []
    for task in TASKS:
        b = base_score(task)
        if b is None:
            continue
        seeds = data.get(task, {}).get(PRIMARY_METRIC[task])
        if not seeds:
            continue
        ds.append(mean(seeds) - b)
    if ds:
        deltas[model] = ds

with (ROOT / "model_avg_delta_ranked.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Rank", "Model", "Tasks_covered", "Avg_delta_vs_BASE"])
    ranked = sorted(deltas.items(), key=lambda kv: mean(kv[1]), reverse=True)
    for rank, (model, ds) in enumerate(ranked, 1):
        w.writerow([rank, model, len(ds), f"{mean(ds):+.4f}"])


# ---- 7) Compact merge-method scorecard: per task, vs best component & vs joint baseline ----
COMPONENT_NAMES = ["MARK+LRD", "FWEdu+LRD", "SYN+LRD"]


def best_component_mean(task: str) -> tuple[str, float] | None:
    metric = PRIMARY_METRIC[task]
    best = None
    for comp in COMPONENT_NAMES:
        seeds = (seed10.get(comp, {}) or {}).get(task, {}).get(metric)
        if seeds is None:
            seeds = (old3.get(comp, {}) or {}).get(task, {}).get(metric)
        if not seeds:
            continue
        mu = mean(seeds)
        if best is None or mu > best[1]:
            best = (comp, mu)
    return best


with (ROOT / "merge_scorecard.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "Merge_method", "Task", "Metric",
        "Merge_mean", "Best_component", "Best_component_mean",
        "Delta_vs_best_component",
        "Joint_FWEdu+MARK+SYN+LRD", "Delta_vs_joint",
        "Joint_FWEdu+MARK+SYN+ZYDA+LRD", "Delta_vs_joint_ZYDA",
    ])
    for method, mdata in merge.items():
        for task in TASKS:
            metric = PRIMARY_METRIC[task]
            mvals = mdata.get(task, {}).get(metric)
            if not mvals:
                continue
            mu_m = mean(mvals)
            bc = best_component_mean(task)
            joint_a = old3.get("FWEdu+MARK+SYN+LRD", {}).get(task, {}).get(metric)
            joint_b = old3.get("FWEdu+MARK+SYN+ZYDA+LRD", {}).get(task, {}).get(metric)
            mu_ja = mean(joint_a) if joint_a else None
            mu_jb = mean(joint_b) if joint_b else None
            w.writerow([
                method, task, metric,
                f"{mu_m:.4f}",
                bc[0] if bc else "",
                f"{bc[1]:.4f}" if bc else "",
                f"{mu_m - bc[1]:+.4f}" if bc else "",
                f"{mu_ja:.4f}" if mu_ja is not None else "",
                f"{mu_m - mu_ja:+.4f}" if mu_ja is not None else "",
                f"{mu_jb:.4f}" if mu_jb is not None else "",
                f"{mu_m - mu_jb:+.4f}" if mu_jb is not None else "",
            ])


# ---- Console summary ----
print("\n=== Summary of n=10 ablation tests ===\n")
with (ROOT / "ablation_n10_stats.csv").open() as f:
    for line in f:
        print(line.rstrip())

print("\n=== Top 15 models by avg delta vs BASE ===\n")
with (ROOT / "model_avg_delta_ranked.csv").open() as f:
    for i, line in enumerate(f):
        if i > 15:
            break
        print(line.rstrip())

print("\nWrote: summary_per_model_n10.csv, ablation_n10_stats.csv, "
      "merge_vs_components.csv, merge_vs_joint.csv, "
      "per_task_ranking.csv, model_avg_delta_ranked.csv")
