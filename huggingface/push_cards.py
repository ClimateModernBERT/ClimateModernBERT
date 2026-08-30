#!/usr/bin/env python3
"""Render ClimateModernBERT model cards from the generated manifest.

Preview is the default and touches nothing. Uploading requires --push, a
`huggingface-cli login` session, and write access to the repository.

    python huggingface/push_cards.py sraj/Merge_Linear
    python huggingface/push_cards.py --all > /tmp/cards.txt
    python huggingface/push_cards.py sraj/Merge_Linear --push

The manifest is generated from site/src/data/models.ts:

    cd site && npx tsx ../scripts/generate-docs.ts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifests" / "models.json"
TEMPLATE = HERE / "model-card-template.md"

STAGE_LABEL = {
    "phase1": "Phase 1 · Continued Pretraining",
    "phase2": "Phase 2 · LRD Specialization",
}

AMBIGUITY_NOTE = (
    "> **Mapping under review.** This checkpoint's correspondence to a configuration in "
    "the manuscript is not fully settled; see the *Open questions* section of "
    "[docs/model-naming.md](https://github.com/Michaelyya/ClimateModernBERT/blob/main/docs/model-naming.md).\n"
)

RECOMMENDED_ID = "CMB-ClimateModernBERT/Merge_Soup_LRD"
LEGACY_RECOMMENDED_ID = "sraj/Merge_Linear"


def load() -> dict:
    if not MANIFEST.exists():
        sys.exit(
            f"missing {MANIFEST}\nRegenerate it with:  cd site && npx tsx ../scripts/generate-docs.ts"
        )
    return json.loads(MANIFEST.read_text())


def canonical_id(m: dict) -> str:
    """The id the card should present: the org name once republished, else sraj."""
    return m.get("new_id") or m["hf_id"]


def render(m: dict, template: str, license_id: str | None) -> str:
    repo = canonical_id(m)
    extra_tags = []
    if m["family"] == "merged":
        extra_tags += ["- merge", "- mergekit"]
    for key, tag in (("academic", "- academic-text"), ("web", "- web-text"), ("synthetic", "- synthetic-data")):
        if m.get(key):
            extra_tags.append(tag)

    merge_rows = ""
    if m.get("merge_method"):
        merge_rows += f"| **Merge method** | {m['merge_method']} |\n"
    components = m.get("merge_components_new") or m.get("merge_components")
    if components:
        links = ", ".join(f"[`{c}`](https://huggingface.co/{c})" for c in components)
        merge_rows += f"| **Merged from** | {links} |\n"
    if m.get("synthetic_generator"):
        merge_rows += f"| **Synthetic generator** | {m['synthetic_generator']} |\n"
    if m.get("paper_variant"):
        merge_rows += f"| **Paper notation** | {m['paper_variant']} |\n"
    if m.get("superseded_by"):
        sb = m["superseded_by"]
        merge_rows += f"| **Superseded by** | [`{sb}`](https://huggingface.co/{sb}) |\n"

    notes = m["notes"] + "\n"
    if m.get("ambiguous"):
        notes += "\n" + AMBIGUITY_NOTE

    # Republished checkpoints must name where the weights came from.
    if m.get("new_id"):
        legacy = m["hf_id"]
        provenance = (
            f"Republished from [`{legacy}`](https://huggingface.co/{legacy}) under a name that "
            "matches the paper's notation. The weights are identical; the original repository "
            "remains available."
        )
    else:
        provenance = ""

    if m.get("avg_f1") is not None:
        eval_block = (
            f"This checkpoint reaches **{m['avg_f1']:.1f} average F1** across the nine "
            "benchmarks, as reported in the manuscript."
        )
    else:
        eval_block = (
            "The manuscript reports no aggregate score for this checkpoint. It is released "
            "for provenance and follow-up work, not as a headline model."
        )

    if repo == RECOMMENDED_ID:
        recommendation = (
            "This is the recommended ClimateModernBERT checkpoint for general use."
        )
    else:
        recommendation = (
            f"For general use, prefer [`{RECOMMENDED_ID}`](https://huggingface.co/{RECOMMENDED_ID}), "
            "the merged model that reaches 76.3 average F1."
        )

    license_note = (
        f"Released under `{license_id}`."
        if license_id
        else (
            "Not yet set on this repository. The upstream base model is "
            "[ModernBERT-Base](https://huggingface.co/answerdotai/ModernBERT-base); check its "
            "terms, and those of the underlying corpora, before redistributing. "
            "No license is asserted here on the maintainers' behalf."
        )
    )

    out = template
    for key, value in {
        "LICENSE_YAML": f"license: {license_id}\n" if license_id else "",
        "EXTRA_TAGS": "\n".join(extra_tags),
        "DISPLAY_NAME": m["display_name"],
        "HF_ID": repo,
        "CORPUS_PHRASE": m["corpus_phrase"] or "Unverified",
        "PAPER_NOTATION": m["paper_notation"] or "not mapped",
        "STAGE_LABEL": STAGE_LABEL[m["stage"]],
        "LEGACY_STAGE": m["legacy_stage"],
        "STATUS_LABEL": m["status_label"],
        "MERGE_ROWS": merge_rows.rstrip("\n"),
        "NOTES": notes,
        "PROVENANCE": provenance,
        "EVAL_BLOCK": eval_block,
        "RECOMMENDATION": recommendation,
        "LICENSE_NOTE": license_note,
    }.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", nargs="?", help="e.g. sraj/Merge_Linear")
    ap.add_argument("--all", action="store_true", help="render every card in the manifest")
    ap.add_argument("--push", action="store_true", help="upload to the Hub (needs auth + write access)")
    ap.add_argument("--license", help="license id to stamp; omitted entirely if not passed")
    args = ap.parse_args()

    if not args.repo and not args.all:
        ap.error("give a repo id or --all")
    if args.push and args.all:
        ap.error("--push takes one repo at a time; uploading 56 cards in a loop is not a thing to do by accident")

    data = load()
    template = TEMPLATE.read_text()
    index = {m["hf_id"]: m for m in data["models"]}
    index.update({m["new_id"]: m for m in data["models"] if m.get("new_id")})

    targets = data["models"] if args.all else [index.get(args.repo)]
    if targets[0] is None:
        sys.exit(f"{args.repo} is not in the manifest. Known ids:\n  " + "\n  ".join(sorted(index)))

    for m in targets:
        card = render(m, template, args.license)
        if args.push:
            from huggingface_hub import HfApi  # imported late so preview needs no dependency

            HfApi().upload_file(
                path_or_fileobj=card.encode(),
                path_in_repo="README.md",
                repo_id=m["hf_id"],
                repo_type="model",
                commit_message="Add model card",
            )
            print(f"pushed README.md to {m['hf_id']}")
        else:
            print(f"{'=' * 78}\n# {m['hf_id']}\n{'=' * 78}\n{card}")


if __name__ == "__main__":
    main()
