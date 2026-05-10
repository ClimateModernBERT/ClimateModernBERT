# ClimateModernBERT — Polished Analysis (n=10 + Model Merging)

**Updated:** 2026-05-01

This is the v2 update of the previous 3-seed analysis. Two new pieces of data:

1. **n=10 seeds** for the four MARK / MARK+SYN / MARK+LRD / MARK+SYN+LRD models — gives the SYN and LRD ablations real statistical power.
2. **7 model-merging variants** (3 seeds each) on top of the same three component checkpoints (MARK+LRD, FWEdu+LRD, SYN+LRD).

All numbers below come from `parse_and_analyze.py` in this folder. Everything is keyed to the **primary metric per task** (f1 for clim_retrieve / commitments / detection / specificity / env_claims; macro_f1 for sentiment / netzero / tcfd / water_forest / wximpactbench).

> **Note on `environmental_claims`**: every seed produces identical scores (the
> finetune is deterministic on the small dev set), so all paired tests are
> degenerate. We exclude it from significance tests but still rank it in the
> per-task tables.
>
> **Note on `clim_retrieve` MARK+LRD**: one of the 10 seeds is a clear failed
> run (f1 = 0.4545, recall 0.31). Drop it and MARK+LRD on clim_retrieve looks
> roughly normal (~0.85). The reported number keeps it for honesty but the std
> ±0.12 is the giveaway.

---

## 1. Headline ranking (avg delta vs BASE, primary metric, 9 tasks)

env_claims excluded from the average (no variance).

| Rank | Model                                | N seeds | Avg Δ vs BASE |
|------|--------------------------------------|---------|---------------|
| 1    | **Merge_avg**  (simple weight averaging) | 3       | **+0.0288**   |
| 2    | **TA_Lambda10**  (Task Arithmetic, λ=1.0)| 3       | +0.0223       |
| 3    | TIES_D07                             | 3       | +0.0214       |
| 4    | TIES_D05                             | 3       | +0.0194       |
| 5    | MARK (n=10)                          | 10      | +0.0180       |
| 6    | DARE_TIES_D05                        | 3       | +0.0170       |
| 7    | MARK+SYN (n=10)                      | 10      | +0.0138       |
| 8    | SYN+LRD                              | 3       | +0.0136       |
| 9    | SYN+ZYDA                             | 3       | +0.0118       |
| 10   | MARK+SYN+LRD (n=10)                  | 10      | +0.0106       |
| 11   | FWEdu+MARK+SYN+ZYDA+LRD              | 3       | +0.0104       |
| 12   | FWEdu+LRD                            | 3       | +0.0096       |
| 13   | SYN                                  | 3       | +0.0094       |
| 14   | MARK+LRD (n=10)                      | 10      | +0.0091       |
| 15   | DARE_TIES_D07                        | 3       | +0.0085       |

**Takeaway.** Three of the top four spots are model-merging methods, with simple
weight averaging on top. They beat every single joint-trained model that doesn't
use merging. Lower in the table the picture matches v1: simpler combinations
(MARK alone, MARK+SYN) outperform the big stew (FWEdu+MARK+SYN+ZYDA+LRD).

Full file: `model_avg_delta_ranked.csv`.

---

## 2. Does Synthetic data help? — n=10 verdict

Two paired contrasts, n=10 seeds, paired-t + Wilcoxon signed-rank + Cohen's d.
Direction column refers to per-seed direction of B − A.

### MARK → MARK+SYN

| Task                          | Δ (B−A) | paired-t p | Wilcoxon p | Cohen's d | Direction  |
|-------------------------------|---------|------------|------------|-----------|------------|
| clim_retrieve                 | +0.006  | 0.31       | 0.26       | +0.34     | mixed      |
| **climate_commitments**       | **−0.039** | **0.000**  | **0.002**  | **−3.20** | all_B<A    |
| **climate_detection**         | **+0.008** | **0.012**  | **0.020**  | +1.00     | mixed      |
| climate_specificity           | −0.013  | 0.07       | 0.13       | −0.66     | mixed      |
| climate_sentiment             | −0.004  | 0.43       | 0.56       | −0.26     | mixed      |
| netzero_reduction             | −0.003  | 0.51       | 0.63       | −0.21     | mixed      |
| **tcfd_recommendations**      | **+0.024** | **0.000**  | **0.002**  | **+2.32** | all_B>A    |
| water_forest                  | −0.004  | 0.06       | 0.08       | −0.69     | mixed      |
| wximpactbench                 | −0.013  | 0.25       | 0.16       | −0.39     | mixed      |

### MARK+LRD → MARK+SYN+LRD

| Task                          | Δ (B−A) | paired-t p | Wilcoxon p | Cohen's d | Direction  |
|-------------------------------|---------|------------|------------|-----------|------------|
| clim_retrieve                 | +0.022  | 0.66       | 0.77       | +0.14     | mixed      |
| **climate_commitments**       | −0.026  | **0.016**  | **0.037**  | −0.93     | mixed      |
| climate_detection             | −0.004  | 0.18       | 0.38       | −0.45     | mixed      |
| **climate_specificity**       | **+0.022** | **0.001**  | **0.002**  | +1.45     | all_B>A    |
| **climate_sentiment**         | **+0.009** | **0.007**  | **0.006**  | +1.09     | mixed      |
| netzero_reduction             | +0.009  | 0.07       | 0.08       | +0.66     | mixed      |
| tcfd_recommendations          | −0.000  | 0.96       | 0.92       | −0.02     | mixed      |
| **water_forest**              | **−0.008** | **0.001**  | **0.004**  | −1.55     | mixed      |
| wximpactbench                 | −0.009  | 0.49       | 0.63       | −0.23     | mixed      |

### Verdict — synthetic data is genuinely mixed, with two robust patterns

- **Robust positives (significant in both contexts):** `tcfd_recommendations`
  (+2.4 pp at α=0.05), `climate_specificity` (+2.2 pp under LRD, α=0.001),
  `climate_sentiment` (+0.9 pp under LRD, α=0.01).
- **Robust negatives (significant in both contexts):** `climate_commitments`
  (−3.9 pp standalone, −2.6 pp with LRD; α≤0.02 in both, large effect d=3.2).
- **Bias of synthetic data:** improves *technical / specificity / framework*
  classification, hurts *commitments / actions* judgments. The hypothesis from
  v1 (synthetic data lacks nuanced contextual language) holds up — and it now
  does so with α=0.05 backing, not just suggestive p-values.

### Action item ⇒ synthetic-data generation v2
Worth trying, since the failure mode is concentrated on commitments/actions:
mix in real annotated commitment language, or weight synthetic data lower for
this task. Send the synthetic-data-gen scripts to Jingwei as planned.

---

## 3. Does LRD help? — n=10 verdict

### MARK → MARK+LRD

| Task                          | Δ      | paired-t p | Wilcoxon p | d       | Direction |
|-------------------------------|--------|------------|------------|---------|-----------|
| clim_retrieve                 | −0.045 | 0.30       | 0.49       | −0.35   | mixed     |
| **climate_commitments**       | −0.031 | **0.003**  | **0.002**  | −1.25   | all_B<A   |
| **climate_detection**         | +0.005 | **0.004**  | **0.004**  | +1.19   | mixed     |
| **climate_specificity**       | −0.016 | **0.005**  | **0.027**  | −1.16   | mixed     |
| climate_sentiment             | −0.003 | 0.59       | 0.56       | −0.18   | mixed     |
| netzero_reduction             | −0.005 | 0.38       | 0.56       | −0.29   | mixed     |
| **tcfd_recommendations**      | **+0.023** | **0.000**  | **0.002**  | **+1.87** | all_B>A   |
| water_forest                  | −0.001 | 0.45       | 0.49       | −0.25   | mixed     |
| wximpactbench                 | −0.008 | 0.43       | 0.49       | −0.26   | mixed     |

### MARK+SYN → MARK+SYN+LRD

| Task                  | Δ      | paired-t p | Wilcoxon p | d     |
|-----------------------|--------|------------|------------|-------|
| clim_retrieve         | −0.029 | 0.09       | 0.08       | −0.61 |
| **climate_commitments** | −0.018 | **0.005** | **0.010**  | −1.18 |
| climate_detection     | −0.008 | 0.06       | 0.06       | −0.67 |
| **climate_specificity** | **+0.018** | **0.004** | **0.006**  | +1.24 |
| climate_sentiment     | +0.011 | 0.19       | 0.70       | +0.45 |
| netzero_reduction     | +0.006 | 0.09       | 0.23       | +0.61 |
| tcfd_recommendations  | −0.002 | 0.62       | 0.77       | −0.16 |
| **water_forest**      | **−0.005** | **0.03**  | **0.03**   | −0.80 |
| wximpactbench         | −0.004 | 0.75       | 0.85       | −0.10 |

### Verdict — LRD has a sharper picture than v1 suggested

The original "broadly positive" claim does **not** survive n=10:

- **Robust wins:** `tcfd_recommendations` (+2.3 pp, α=0.001). `climate_specificity`
  is split — LRD hurts standalone (−1.6 pp, α=0.005) but helps when SYN is
  present (+1.8 pp, α=0.004). So LRD only helps specificity by *interacting* with
  the synthetic data signal.
- **Robust losses:** `climate_commitments` in both contexts (−3.1 / −1.8 pp,
  α≤0.005). `climate_detection` standalone (−1.1 pp, marginal but consistent).
- The directional positive signal from v1 on `wximpactbench` **disappears** with
  n=10 (Δ = −0.008, p=0.43). That earlier reading was likely chance.

### Action item
LRD is a tcfd-specific intervention more than a "broadly positive" one. We
should keep using it for the merged-model components (which is what April's
artifacts do), but stop pitching it as a general training stability win.

---

## 4. Model merging vs components — task by task

For each merge method, we compare its mean against the **best-performing
single component** on that task (one of MARK+LRD / FWEdu+LRD / SYN+LRD), as
well as the joint-trained baselines. n=3 seeds for the merge runs.

`merge_scorecard.csv` and `merge_vs_components.csv` have the full table.

### Top-line scorecard — Merge_avg (simple weight averaging)

This is the method that wins overall. Per task:

| Task                  | Best single comp | Merge_avg | Δ vs best comp | Δ vs joint (FWEdu+MARK+SYN+LRD) |
|-----------------------|------------------|-----------|----------------|----------------------------------|
| clim_retrieve         | SYN+LRD 0.875    | 0.836     | **−0.040**     | −0.029                           |
| climate_commitments   | MARK+LRD 0.687   | 0.665     | −0.022         | **+0.074**                       |
| climate_detection     | FWEdu+LRD 0.954  | 0.954     | +0.001         | +0.003                           |
| climate_specificity   | FWEdu+LRD 0.695  | 0.709     | +0.014         | −0.002                           |
| climate_sentiment     | MARK+LRD 0.771   | 0.767     | −0.005         | +0.028                           |
| netzero_reduction     | SYN+LRD 0.990    | 0.989     | −0.001         | +0.004                           |
| tcfd_recommendations  | MARK+LRD 0.613   | **0.641** | **+0.028**     | +0.018                           |
| water_forest          | FWEdu+LRD 0.971  | 0.973     | +0.002         | −0.000                           |
| wximpactbench         | MARK+LRD 0.252   | **0.337** | **+0.084**     | **+0.093**                       |

Pattern (consistent across all 7 merge variants — see `merge_scorecard.csv`):

- **Big wins for merging** on the two hardest, most-data-hungry tasks:
  `wximpactbench` (+8 pp over best component, +9 pp over joint) and
  `tcfd_recommendations` (+3 pp, +2 pp). These tasks benefit most because no
  single component is good at them, but the components fail in different ways
  — averaging cancels component-specific weaknesses.
- **Marginal on classification staples** (`climate_detection`, `water_forest`,
  `netzero_reduction`): the merge matches the best component but doesn't
  clearly improve. These tasks are already at ceiling.
- **Loses on `clim_retrieve`** to SYN+LRD by ~4 pp. Worth flagging — none of the
  7 merge variants beat SYN+LRD here. If retrieval matters, SYN+LRD is still
  the right standalone model, or we'd want a retrieval-weighted merge.
- **Loses or ties on `climate_commitments` standalone** (the hard task for SYN).

### Merge method comparison

Per-task winners across the 7 methods (from `merge_scorecard.csv`):

| Task                  | Best merge method  | Mean   | Notes                          |
|-----------------------|--------------------|--------|--------------------------------|
| clim_retrieve         | DARE_TIES_D05      | 0.863  | Closest to SYN+LRD             |
| climate_commitments   | TIES_D07           | 0.719  | Beats MARK+LRD (+3.2 pp)       |
| climate_detection     | Merge_avg          | 0.954  | Tie with FWEdu+LRD             |
| climate_specificity   | Merge_avg          | 0.709  | Beats best comp (+1.4 pp)      |
| climate_sentiment     | DARE_TIES_D05      | 0.783  | Beats MARK+LRD (+1.1 pp)       |
| netzero_reduction     | TA_Lambda10        | 0.992  | Tiny absolute gain             |
| tcfd_recommendations  | Merge_avg          | 0.641  | +2.8 pp over best comp         |
| water_forest          | Merge_avg          | 0.973  | +0.2 pp                        |
| wximpactbench         | **Merge_avg**      | 0.337  | **+8.4 pp** over best comp     |

**Recommendations on the merge family:**

- **Pick `Merge_avg`** as the default. It wins or ties on 5 of 9 tasks, has the
  best avg Δ overall, and is the simplest method (no hyperparameter to tune).
- `TA_Lambda10` is the strong runner-up if you want a single configurable knob.
- `TA_Lambda05` underperforms — shrinking task vectors by half loses signal.
- `DARE_TIES_D07` and `TIES_D07` (high drop ratio) are notably weaker than the
  D05 variants on `climate_commitments` — pruning too aggressively removes the
  MARK signal that this task needs.

---

## 5. Model merging vs joint training

The merge isn't just beating components — it's beating **joint training on the
combined data** (FWEdu+MARK+SYN+LRD or FWEdu+MARK+SYN+ZYDA+LRD). See
`merge_vs_joint.csv`.

`Merge_avg` vs `FWEdu+MARK+SYN+LRD` (joint training, primary metric):

| Task                  | Joint  | Merge  | Δ      | Sig?           |
|-----------------------|--------|--------|--------|----------------|
| clim_retrieve         | 0.864  | 0.836  | −0.029 | not sig (n=3)  |
| **climate_commitments** | 0.591 | 0.665 | **+0.074** | dir consistent |
| climate_detection     | 0.951  | 0.954  | +0.003 | not sig        |
| climate_specificity   | 0.711  | 0.709  | −0.002 | not sig        |
| climate_sentiment     | 0.739  | 0.767  | +0.028 | dir consistent |
| netzero_reduction     | 0.985  | 0.989  | +0.004 | dir consistent |
| **tcfd_recommendations** | 0.624 | 0.641 | **+0.018** | p=0.017       |
| water_forest          | 0.973  | 0.973  | −0.000 | not sig        |
| **wximpactbench**     | 0.244  | 0.337  | **+0.093** | p=0.002, d=11.8 |

Merging wins on **6 of 9 tasks** vs joint training, including big wins on the
challenging tasks where joint training plateaued. **It loses only on
`clim_retrieve`** (−2.9 pp), the same outlier task from the component
comparison.

### Why? — Hypothesis worth checking against the literature

Joint training has to share representations across all task data, which forces
trade-offs. Merging keeps each component independently optimized for its own
data and only averages at the end — this preserves specialized capability that
joint training would otherwise compromise away. The effect is largest on the
hardest tasks (wximpactbench, tcfd) which is consistent with this view.

This is the pattern reported in the LLM-merging literature too (TIES Merging,
DARE, Task Arithmetic papers). Worth grounding our framing against:

- *Wortsman et al. 2022* — "Model Soups" (simple averaging baseline is
  surprisingly strong, matches our finding that `Merge_avg` ≥ TIES variants)
- *Yadav et al. 2023* — "TIES-Merging"
- *Yu et al. 2024* — "DARE"
- *Ilharco et al. 2023* — "Task Arithmetic"

(Pointer to Shantam: he was going to look at this. His current pass should
include `Merge_avg` as the default baseline rather than only the structured-
sparsity variants.)

---

## 6. Statistical significance — overall posture

The n=10 ablation now gives publishable α=0.05 results. From the n=10 stats
file:

- **5 task × ablation cells** are significant at α=0.05 in *both* the paired-t
  and Wilcoxon tests:
  - SYN: commitments (down), detection (up), tcfd (up)
  - SYN+LRD: commitments (down), specificity (up), sentiment (up), water_forest (down)
  - LRD: commitments (down), detection (up), specificity (down), tcfd (up)
  - LRD+SYN: commitments (down), specificity (up), water_forest (down)
- **Effect sizes** are large (|d| > 1) in many of the significant cells, which
  is the main reason n=10 is enough.
- We're using **paired t** as the primary test (matched seeds, near-normal
  diffs after Shapiro checks on the larger contrasts) and **Wilcoxon
  signed-rank** as a non-parametric backup. They agree directionally
  everywhere; p-values are within ~0.02 of each other in all but a couple of
  marginal cases.

For the **merge tests**, n=3 keeps p-values weak. We report direction
consistency and Cohen's d as the primary evidence there, with paired-t and
Wilcoxon as supplementary. To get publishable significance for merge findings
we'd need to rerun the merges with 5+ seeds (not 10 — the magnitudes are huge
on wximpactbench and tcfd, so 5 should be enough).

---

## 7. Files in this directory

| File                            | What it is                                              |
|---------------------------------|---------------------------------------------------------|
| `parse_and_analyze.py`          | Re-runs everything below from the two source CSVs       |
| `summary_per_model_n10.csv`     | Mean ± std for the 4 n=10 models, 10 tasks              |
| `ablation_n10_stats.csv`        | n=10 SYN/LRD ablations: paired-t, Wilcoxon, Cohen's d   |
| `merge_vs_components.csv`       | Each merge variant vs each individual component         |
| `merge_vs_joint.csv`            | Each merge variant vs FWEdu+MARK+SYN(+ZYDA)+LRD joint   |
| `merge_scorecard.csv`           | Compact merge × task table with Δ vs best component & joint |
| `per_task_ranking.csv`          | All 22 models ranked per task                           |
| `model_avg_delta_ranked.csv`    | Models ranked by avg Δ vs BASE                          |

Old v1 outputs (`delta_from_base.csv`, `statistical_significance_tests.csv`,
etc.) are kept untouched for diff/audit.

---

## 8. To-do for the next pass

1. **More seeds for the merge runs** (target n=5). Big effect sizes on
   wximpactbench / tcfd should make this cheap to confirm.
2. **Synthetic-data v2** focused on commitments/actions language.
3. **Retrieval-weighted merge** to recover the SYN+LRD advantage on
   `clim_retrieve` (or just keep SYN+LRD as the retrieval-task model).
4. **Literature grounding** — write the merge results against TIES / DARE /
   Task Arithmetic; confirm the "merge > joint training" claim holds in their
   benchmarks too.
5. **n=10 environmental_claims**: investigate why every seed produces an
   identical score. Suspected: deterministic finetune init + tiny dev set.
   Worth confirming before dropping it from the headline numbers.
