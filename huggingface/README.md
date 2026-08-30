# Hugging Face release kit

Content for the ClimateModernBERT model cards. **Nothing here is pushed automatically.**
Every published checkpoint under [`sraj`](https://huggingface.co/sraj) currently has *no*
`README.md`, so the Hub renders a bare file listing and none of them explain what corpus
they were trained on.

| File | What it is |
|---|---|
| `model-card-template.md` | The card body. Placeholders in `{{DOUBLE_BRACES}}`. |
| `manifests/models.json` | Generated manifest — the values to substitute, for all 56 checkpoints. |
| `push_cards.py` | Renders a card for one repo and prints it, or uploads it with `--push`. |

## Filling in a card

Each entry in `manifests/models.json` carries everything the template needs:
`display_name`, `paper_notation`, `corpus_phrase`, `stage`, `legacy_stage`, `avg_f1`,
`merge_method`, `merge_components`, `status`, `notes`.

```bash
# Preview (writes nothing, contacts nothing)
python huggingface/push_cards.py sraj/Merge_Linear

# Preview every card
python huggingface/push_cards.py --all > /tmp/cards.txt

# Upload — requires `huggingface-cli login` and write access to the repo
python huggingface/push_cards.py sraj/Merge_Linear --push
```

## Before pushing anything

- **Licenses.** The template leaves `license` out of the YAML front matter unless you
  pass `--license`. Do not assign one on a repository whose licensing has not been
  confirmed — an absent license field is more honest than a guessed one.
- **The manuscript is unpublished.** The template says "under review" and gives no
  venue, DOI or BibTeX. Keep it that way until the paper is public.
- **Two mappings are unresolved.** See the *Open questions* section of
  [`docs/model-naming.md`](../docs/model-naming.md). Cards for the affected checkpoints
  render a "mapping under review" note; resolve the question before removing it.
- **Do not claim every checkpoint is the best.** The template states each model's own
  reported average F1 and points at the recommended model where it differs.
