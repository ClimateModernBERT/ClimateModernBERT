---
{{LICENSE_YAML}}library_name: transformers
pipeline_tag: fill-mask
base_model: answerdotai/ModernBERT-base
language:
- en
tags:
- climate
- modernbert
- domain-adaptation
- continued-pretraining
{{EXTRA_TAGS}}
---

# {{DISPLAY_NAME}}

Part of **ClimateModernBERT**, a family of climate-domain encoders obtained by continued
pretraining of ModernBERT-Base on climate text.

| | |
|---|---|
| **Repository** | `{{HF_ID}}` |
| **Naming** | corpora in `A_S_F` order, then the training stage: `CX` = Phase 1, `CX_LRD` = Phase 1 + Phase 2 |
| **Corpora** | {{CORPUS_PHRASE}} — {{PAPER_NOTATION}} in the paper's notation |
| **Training stage** | {{STAGE_LABEL}} (legacy suffix `{{LEGACY_STAGE}}`) |
| **Base model** | ModernBERT-Base, pre-LRD stable-phase checkpoint |
| **Architecture** | 150M parameters · 22 layers · hidden 768 · 12 heads · vocab 50,368 · 8,192-token context |
| **Status** | {{STATUS_LABEL}} |
{{MERGE_ROWS}}

{{NOTES}}
{{PROVENANCE}}

## Training data

Continued pretraining used {{CORPUS_PHRASE}} from a 6.42B-token climate corpus:

| | Corpus | Tokens | Description |
|---|---|---|---|
| 𝒜 | Academic | ~1.28B | Peer-reviewed journal articles across climate science, earth systems and energy economics; the ClimateNews archive 2000–2022; climate arXiv preprints; climate handbooks. |
| ℱ | Climate Web | ~5B | FineWeb-Edu filtered for climate relevance with a 166-term keyword filter followed by a FastText classifier. |
| 𝒮 | Synthetic | ~0.14B | LLM-generated climate text conditioned on in-domain seed excerpts, in three communication styles. |

Raw academic text is not redistributed: peer-reviewed articles are accessed under
institutional publisher licenses, and news shards and handbooks were collected for
non-commercial research use. The processing pipelines are released instead.

## Training procedure

Two stages, following ModernBERT's own continued-pretraining recipe:

- **Phase 1 — context extension.** 3 epochs, constant LR 3e-4, global batch 576,
  sequence length 8,192, MLM masking 30%, StableAdamW, BF16.
- **Phase 2 — LRD specialization.** 3 further epochs on a `1 − √t` decay schedule from
  LR 3e-4 with final LR factor 1e-3.

4× NVIDIA A100, MosaicML Composer. Final checkpoints are converted to HF Transformers format.

## Usage

ModernBERT is native to `transformers` from **4.48** onward, so no `trust_remote_code`
is required.

```python
from transformers import AutoTokenizer, AutoModel

model_id = "{{HF_ID}}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

inputs = tokenizer("Scope 1 and 2 emissions fell 12% against a 2019 baseline.", return_tensors="pt")
outputs = model(**inputs)   # outputs.last_hidden_state -> (batch, seq, 768)
```

For a downstream task, load `AutoModelForSequenceClassification` and fine-tune. The
paper's recipe: LR 4e-5, effective batch 64, weight decay 0.01, up to 10 epochs with
early stopping on validation F1, BF16 with fused AdamW.

## Evaluation

{{EVAL_BLOCK}}

Evaluated on nine climate NLP benchmarks: Climate Detection, Climate Specificity,
Commitments & Actions, Climate Sentiment, Net Zero & Reduction, TCFD Recommendations,
WFB Nature, WXImpactBench, and ClimRetrieve. Binary tasks and ClimRetrieve report
positive-class F1; multi-class and multi-label tasks report macro-F1. Scores are the mean
over three fine-tuning seeds under a single shared hyperparameter configuration.

Reference points from the paper: the ModernBERT-Base stable-phase baseline reaches 73.5
average F1, and ClimateBERT reaches 72.1 under the same protocol.

{{RECOMMENDATION}}

## Intended use

Research on climate NLP: encoding climate text, and fine-tuning for classification,
multi-label tagging, and retrieval over corporate disclosures, policy documents,
scientific literature and climate news.

## Limitations

- English only, and built on a single encoder family (ModernBERT-Base).
- Current climate NLP benchmarks are largely sentence- or passage-level, so the model's
  long-context capacity is not fully exercised by the reported evaluation.
- Findings about corpus composition are demonstrated within climate NLP and should not be
  read as universal principles of domain adaptation.
- Synthetic training data has task-dependent effects: it helps taxonomy- and
  framework-driven tasks while degrading performance on tasks requiring finer-grained
  discourse and commitment understanding.
- The model is a masked language model, not an instruction-following system, and produces
  no calibrated factual guarantees about climate science.

## Paper

**Climate-ModernBERT: Revisiting Corpus Composition for Domain-Adaptive Continued
Pretraining.** Preprint manuscript, currently under review — no venue, DOI or arXiv
identifier yet, and no citation to give. The PDF is hosted in the project repository.

- Project website: https://michaelyya.github.io/ClimateModernBERT/
- Code and pipelines: https://github.com/Michaelyya/ClimateModernBERT
- Full model catalog: https://github.com/Michaelyya/ClimateModernBERT/blob/main/docs/model-inventory.md
- Naming guide: https://github.com/Michaelyya/ClimateModernBERT/blob/main/docs/model-naming.md

## License

{{LICENSE_NOTE}}
