#!/usr/bin/env python3
"""Republish the paper's checkpoints under the CMB-ClimateModernBERT org.

The legacy repositories live under `sraj/*` and this account cannot write to
them, so nothing is renamed or deleted remotely: weights are downloaded and
re-uploaded under a name that matches the paper's notation, and every old link
keeps working.

    A_S_F_CX       Phase 1 on academic + synthetic + web
    A_S_F_CX_LRD   the same, plus Phase 2 LRD specialization
    Merge_*        parameter-space merges

The mapping is defined in site/src/data/models.ts and reaches this script via
huggingface/manifests/models.json. Regenerate that first if you change it:

    cd site && npx tsx ../scripts/generate-docs.ts

Usage:
    export HF_TOKEN=...                     # never commit this
    python scripts/migrate_to_org.py                 # dry run: print the plan
    python scripts/migrate_to_org.py --push          # do it
    python scripts/migrate_to_org.py --push --only A_CX Merge_Soup_LRD
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "huggingface"))

import json  # noqa: E402
from push_cards import render as render_card  # noqa: E402

MANIFEST = ROOT / "huggingface" / "manifests" / "models.json"
TEMPLATE = ROOT / "huggingface" / "model-card-template.md"

# pytorch_model.bin duplicates model.safetensors byte-for-byte in the source
# repos; carrying it would double ~15GB of transfer for nothing. README.md is
# regenerated from the manifest, and .gitattributes is created by the Hub.
WANTED = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "mergekit_config.yml",
]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--push", action="store_true", help="actually create and upload (default: dry run)")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="restrict to these new repo names")
    ap.add_argument("--limit", type=int, help="stop after N models")
    ap.add_argument("--private", action="store_true", help="create the repos private")
    ap.add_argument("--license", help="license id to stamp on the cards; omitted if not given")
    ap.add_argument("--cache", default=str(ROOT / ".migration-cache"), help="local download cache")
    args = ap.parse_args()

    data = json.loads(MANIFEST.read_text())
    template = TEMPLATE.read_text()
    todo = [m for m in data["models"] if m.get("new_id")]
    if args.only:
        wanted = set(args.only)
        todo = [m for m in todo if m["new_id"].split("/")[-1] in wanted or m["new_id"] in wanted]
        missing = wanted - {m["new_id"].split("/")[-1] for m in todo} - {m["new_id"] for m in todo}
        if missing:
            sys.exit(f"unknown target name(s): {', '.join(sorted(missing))}")
    if args.limit:
        todo = todo[: args.limit]

    if not todo:
        sys.exit("nothing to migrate")

    print(f"{len(todo)} checkpoint(s) to republish into {data['organization']}\n")
    for m in todo:
        print(f"  {m['new_id'].split('/')[-1]:<24} <- {m['hf_id']}")
    print()

    if not args.push:
        print("Dry run. Re-run with --push to create the repositories and upload.")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN in the environment (do not pass it on the command line)")

    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    api = HfApi(token=token)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    done, skipped, failed = [], [], []
    t0 = time.time()

    for i, m in enumerate(todo, 1):
        src, dst = m["hf_id"], m["new_id"]
        name = dst.split("/")[-1]
        print(f"[{i}/{len(todo)}] {name}")

        try:
            # Which of the wanted files does the source actually have?
            src_files = {f.rfilename for f in api.model_info(src).siblings}
            patterns = [p for p in WANTED if p in src_files]

            # Idempotent: if the destination already holds the same weights, skip.
            try:
                dst_files = {f.rfilename for f in api.model_info(dst).siblings}
                if "model.safetensors" in dst_files:
                    a = api.get_paths_info(src, ["model.safetensors"], repo_type="model")[0]
                    b = api.get_paths_info(dst, ["model.safetensors"], repo_type="model")[0]
                    if getattr(a, "lfs", None) and getattr(b, "lfs", None) and a.lfs.sha256 == b.lfs.sha256:
                        print("      already up to date, refreshing card only")
                        api.upload_file(
                            path_or_fileobj=render_card(m, template, args.license).encode(),
                            path_in_repo="README.md",
                            repo_id=dst,
                            repo_type="model",
                            commit_message="Update model card",
                        )
                        skipped.append(name)
                        continue
            except RepositoryNotFoundError:
                pass

            print(f"      downloading {len(patterns)} file(s) from {src}")
            local = snapshot_download(
                repo_id=src, allow_patterns=patterns, cache_dir=str(cache), token=token
            )
            size = sum(p.stat().st_size for p in Path(local).rglob("*") if p.is_file())

            (Path(local) / "README.md").write_text(render_card(m, template, args.license))

            api.create_repo(dst, repo_type="model", exist_ok=True, private=args.private)
            print(f"      uploading {human(size)} to {dst}")
            api.upload_folder(
                folder_path=local,
                repo_id=dst,
                repo_type="model",
                commit_message=f"Republish {src} as {name}",
                ignore_patterns=["pytorch_model.bin", ".gitattributes", ".cache*"],
            )
            print("      done")
            done.append(name)

        except Exception as e:  # keep going; report at the end
            print(f"      FAILED: {type(e).__name__}: {e}")
            failed.append((name, f"{type(e).__name__}: {e}"))

    mins = (time.time() - t0) / 60
    print(f"\n{'=' * 60}")
    print(f"uploaded {len(done)} · already current {len(skipped)} · failed {len(failed)}   ({mins:.1f} min)")
    if failed:
        print("\nfailures:")
        for name, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
