#!/usr/bin/env python3
"""Upload a local folder to the Hugging Face Hub."""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

_slugify_pattern = re.compile(r"[^a-z0-9-]+")


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = value.replace("_", "-").replace(" ", "-")
    value = _slugify_pattern.sub("-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def _collection_slug_candidates(owner: str, raw_identifier: str) -> list[str]:
    owner = owner.strip()
    identifier = raw_identifier.strip()
    candidates: list[str] = []

    if "/" in identifier:
        parts = identifier.split("/", 1)
        owner_part = parts[0] or owner
        slug_part = parts[1]
        candidates.append(f"{owner_part}/{slug_part}")
        normalized = _slugify(slug_part)
        if normalized and normalized != slug_part:
            candidates.append(f"{owner_part}/{normalized}")
    else:
        candidates.append(f"{owner}/{identifier}")
        normalized = _slugify(identifier)
        if normalized and normalized != identifier:
            candidates.append(f"{owner}/{normalized}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local model folder to the Hugging Face Hub."
    )
    parser.add_argument(
        "--folder-path",
        type=Path,
        required=True,
        help="Path to the local folder that should be uploaded.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target repository on the Hub, e.g. `username/model-name`.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=("model", "dataset", "space"),
        help="Type of repository to upload to.",
    )
    parser.add_argument(
        "--path-in-repo",
        default="",
        help="Optional subdirectory inside the repository to upload into.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Target branch or revision.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Custom commit message. Defaults to `Upload <folder-name>`.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Explicit Hugging Face token. Overrides token-file and environment.",
    )
    parser.add_argument(
        "--token-file",
        default=None,
        help="Path to a file containing the Hugging Face token.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Glob pattern(s) to include. Repeat for multiple patterns.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="Glob pattern(s) to exclude. Repeat for multiple patterns.",
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete files in the repo that are not present locally.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private if it does not exist.",
    )
    parser.add_argument(
        "--collection",
        action="append",
        default=None,
        help="Collection slug to add the uploaded repo to. Repeat for multiple collections.",
    )
    parser.add_argument(
        "--collection-owner",
        default=None,
        help="Owner (user or org) for collections when slugs omit the namespace. Defaults to repo owner.",
    )
    return parser.parse_args()


def resolve_token(args: argparse.Namespace) -> Optional[str]:
    if args.token:
        return args.token.strip()
    if args.token_file:
        token_path = Path(args.token_file).expanduser().resolve()
        if not token_path.is_file():
            raise FileNotFoundError(f"Token file not found: {token_path}")
        return token_path.read_text(encoding="utf-8").strip()
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token.strip()
    return None


def main() -> int:
    args = parse_args()
    folder_path = args.folder_path.expanduser().resolve()
    if not folder_path.is_dir():
        print(f"Error: folder not found -> {folder_path}", file=sys.stderr)
        return 1

    try:
        token = resolve_token(args)
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    if not token:
        print(
            "Error: Hugging Face token missing. Use --token, --token-file, or set HF_TOKEN.",
            file=sys.stderr,
        )
        return 1

    api = HfApi(token=token)
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
            token=token,
        )
    except HfHubHTTPError as err:
        print(f"Error creating repo '{args.repo_id}': {err}", file=sys.stderr)
        return 1

    commit_message = args.commit_message or f"Upload {folder_path.name}"
    print(f"Uploading {folder_path} -> {args.repo_id}", flush=True)

    upload_kwargs = dict(
        folder_path=str(folder_path),
        path_in_repo=args.path_in_repo,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        token=token,
        commit_message=commit_message,
    )

    if args.allow_pattern:
        upload_kwargs["allow_patterns"] = args.allow_pattern
    if args.ignore_pattern:
        upload_kwargs["ignore_patterns"] = args.ignore_pattern
    if args.delete_existing:
        upload_kwargs["delete_existing"] = True

    try:
        api.upload_folder(**upload_kwargs)
    except Exception as err:  # huggingface_hub may raise various errors
        print(f"Error uploading to '{args.repo_id}': {err}", file=sys.stderr)
        return 1

    if args.collection:
        collection_owner = args.collection_owner or args.repo_id.split("/", 1)[0]
        for slug in args.collection:
            candidate_slugs = _collection_slug_candidates(collection_owner, slug)
            last_error: Optional[Exception] = None
            for candidate in candidate_slugs:
                try:
                    api.add_collection_item(
                        collection_slug=candidate,
                        item_id=args.repo_id,
                        item_type=args.repo_type,
                        token=token,
                    )
                    print(f"Added {args.repo_id} to collection {candidate}", flush=True)
                    last_error = None
                    break
                except HfHubHTTPError as err:
                    # Keep trying next candidate
                    last_error = err
            if last_error is not None:
                print(
                    f"Warning: could not add {args.repo_id} to collection '{slug}': {last_error}",
                    file=sys.stderr,
                )

    print(f"Upload complete for {args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())