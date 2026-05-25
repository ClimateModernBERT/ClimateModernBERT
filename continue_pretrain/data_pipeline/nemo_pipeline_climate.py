"""
NeMo Curator Pretraining Data Pipeline
=======================================
Pipeline stages (run in order):
  1. Text Cleaning  — Unicode fix, newline normalization, URL removal
  2. Exact Duplicate Removal — MD5-based hashing
  3. Fuzzy Duplicate Removal — MinHash + LSH

Requirements:
  - Ray cluster (GPU support recommended for dedup at scale)
  - nemo-curator package installed

References:
  - https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/content-processing/text-cleaning.html
  - https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/exact.html
  - https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/fuzzy.html
"""

import os
import json
import shutil
import argparse
import glob

import pyarrow.parquet as pq

# --- Text cleaning imports ---
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modifiers import (
    UnicodeReformatter,
    UrlRemover,
    NewlineNormalizer,
)
from nemo_curator.stages.text.modules import Modify

# --- Deduplication imports ---
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
from nemo_curator.stages.text.deduplication.removal_workflow import (
    TextDuplicatesRemovalWorkflow,
)

# Maximum size (in bytes) for a single input JSONL file before splitting
MAX_SINGLE_FILE_BYTES = 500 * 1024 * 1024  # 500 MiB

# Spurious index column that cuDF/dask sometimes writes to parquet partitions
_INDEX_COL = "__index_level_0__"


def split_large_jsonl_files(input_dir: str, max_bytes: int = MAX_SINGLE_FILE_BYTES) -> str:
    """Split any JSONL file larger than max_bytes into smaller chunks.

    If splitting occurs, returns the path to a new directory containing the
    split files.  Otherwise returns the original input_dir unchanged.
    """
    files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
    needs_split = any(os.path.getsize(f) > max_bytes for f in files)
    if not needs_split:
        return input_dir

    split_dir = input_dir.rstrip("/") + "_split"
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir, exist_ok=True)

    for fpath in files:
        if os.path.getsize(fpath) <= max_bytes:
            shutil.copy2(fpath, split_dir)
            continue

        # Split large file into ~max_bytes chunks
        base = os.path.splitext(os.path.basename(fpath))[0]
        chunk_idx = 0
        out_fh = None
        written = 0
        try:
            with open(fpath, "r", encoding="utf-8") as src:
                for line in src:
                    if out_fh is None or written >= max_bytes:
                        if out_fh is not None:
                            out_fh.close()
                        chunk_name = f"{base}_chunk{chunk_idx:04d}.jsonl"
                        out_fh = open(os.path.join(split_dir, chunk_name), "w", encoding="utf-8")
                        chunk_idx += 1
                        written = 0
                    out_fh.write(line)
                    written += len(line.encode("utf-8"))
        finally:
            if out_fh is not None:
                out_fh.close()

    n_out = len(glob.glob(os.path.join(split_dir, "*.jsonl")))
    print(f"  Split {len(files)} input file(s) into {n_out} chunks in {split_dir}")
    return split_dir


def normalize_parquet_schemas(parquet_dir: str) -> None:
    """Remove spurious __index_level_0__ column from parquet files.

    cuDF/dask sometimes writes the DataFrame index as an extra column in
    some partitions but not others.  When downstream stages (e.g. LSH in
    fuzzy dedup) try to ``cudf.read_parquet()`` over all files at once,
    mismatched column counts cause a RuntimeError.  This helper rewrites
    any affected files in-place so all schemas are consistent.

    It also fixes the embedded pandas metadata so that ``index_columns``
    no longer references the deleted column (cuDF consults this metadata
    and will crash with ``invalid schema_idx`` if the column is gone but
    still referenced).
    """
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    fixed = 0
    for fpath in parquet_files:
        schema = pq.read_schema(fpath)
        has_phys_col = _INDEX_COL in schema.names
        # Also check if pandas metadata references the index column
        has_meta_ref = False
        if schema.pandas_metadata:
            idx_cols = schema.pandas_metadata.get("index_columns", [])
            has_meta_ref = _INDEX_COL in idx_cols

        if not has_phys_col and not has_meta_ref:
            continue

        table = pq.read_table(fpath)
        if has_phys_col:
            table = table.drop(_INDEX_COL)

        # Fix pandas metadata: replace index_columns reference to
        # the deleted column with a simple RangeIndex descriptor.
        pandas_meta = schema.pandas_metadata
        if pandas_meta and _INDEX_COL in pandas_meta.get("index_columns", []):
            pandas_meta["index_columns"] = [
                {"kind": "range", "name": None,
                 "start": 0, "stop": len(table), "step": 1}
            ]
            # Also remove the column entry from the columns list
            pandas_meta["columns"] = [
                c for c in pandas_meta.get("columns", [])
                if c.get("name") != _INDEX_COL
            ]
            new_metadata = dict(table.schema.metadata or {})
            new_metadata[b"pandas"] = json.dumps(pandas_meta).encode()
            table = table.replace_schema_metadata(new_metadata)

        pq.write_table(table, fpath)
        fixed += 1
    if fixed:
        print(f"  Normalized {fixed}/{len(parquet_files)} parquet files (dropped {_INDEX_COL})")


# ---------------------------------------------------------------------------
# Stage 1 – Text Cleaning
# ---------------------------------------------------------------------------
def run_text_cleaning(input_dir: str, output_dir: str) -> None:
    """Clean raw text: fix Unicode, normalise newlines, strip URLs."""
    print("=" * 60)
    print("STAGE 1: Text Cleaning")
    print("=" * 60)

    pipeline = Pipeline(
        name="text_cleaning_pipeline",
        description="Clean text: Unicode reformatter, newline normalizer, URL remover",
    )

    pipeline.add_stage(JsonlReader(file_paths=input_dir, blocksize="512MiB"))
    pipeline.add_stage(Modify(UnicodeReformatter()))
    pipeline.add_stage(Modify(NewlineNormalizer()))
    pipeline.add_stage(Modify(UrlRemover()))
    pipeline.add_stage(JsonlWriter(path=output_dir))

    pipeline.run()
    print(f"  -> Text cleaning complete  ->  {output_dir}\n")


# ---------------------------------------------------------------------------
# Stage 2 – Exact Duplicate Removal
# ---------------------------------------------------------------------------
def run_exact_dedup(
    input_dir: str,
    results_dir: str,
    output_dir: str,
    input_filetype: str = "jsonl",
) -> None:
    """Remove character-for-character duplicates via MD5 hashing."""
    print("=" * 60)
    print("STAGE 2: Exact Duplicate Removal")
    print("=" * 60)

    # Step 1 – Identify duplicates
    exact_workflow = ExactDeduplicationWorkflow(
        input_path=input_dir,
        output_path=results_dir,
        text_field="text",
        assign_id=True,
        perform_removal=False,          # removal handled by TextDuplicatesRemovalWorkflow
        input_filetype=input_filetype,
        input_blocksize="512MiB",       # match removal workflow blocksize; smaller to avoid OOM
    )
    exact_workflow.run()

    # Step 2 – Remove duplicates (only if duplicates were actually found)
    id_gen_path = os.path.join(results_dir, "exact_id_generator.json")
    if os.path.exists(id_gen_path):
        print(f"  -> Exact duplicate IDs saved  ->  {results_dir}/ExactDuplicateIds/")
        removal_workflow = TextDuplicatesRemovalWorkflow(
            input_path=input_dir,
            ids_to_remove_path=os.path.join(results_dir, "ExactDuplicateIds"),
            output_path=output_dir,
            input_filetype=input_filetype,
            input_id_field="_curator_dedup_id",
            ids_to_remove_duplicate_id_field="_curator_dedup_id",
            id_generator_path=id_gen_path,
            input_blocksize="512MiB",        # smaller blocksize to avoid OOM on large datasets
        )
        removal_workflow.run()
        print(f"  -> Exact dedup complete  ->  {output_dir}\n")
    else:
        print("  -> No exact duplicates found — copying input to output")
        shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)
        print(f"  -> Exact dedup complete (no removal needed)  ->  {output_dir}\n")


# ---------------------------------------------------------------------------
# Stage 3 – Fuzzy Duplicate Removal
# ---------------------------------------------------------------------------
def run_fuzzy_dedup(
    input_dir: str,
    cache_dir: str,
    results_dir: str,
    output_dir: str,
    input_filetype: str = "jsonl",
) -> None:
    """Remove near-duplicates via MinHash + Locality Sensitive Hashing."""
    print("=" * 60)
    print("STAGE 3: Fuzzy Duplicate Removal")
    print("=" * 60)

    # Step 1 – Identify duplicates
    fuzzy_workflow = FuzzyDeduplicationWorkflow(
        input_path=input_dir,
        cache_path=cache_dir,
        output_path=results_dir,
        text_field="text",
        perform_removal=False,          # removal handled by TextDuplicatesRemovalWorkflow
        input_filetype=input_filetype,
        input_blocksize="512MiB",       # match removal workflow blocksize; smaller to avoid OOM
        char_ngrams=24,
        num_bands=20,
        minhashes_per_band=13,
    )
    fuzzy_workflow.run()

    # Step 2 – Remove duplicates (only if duplicates were actually found)
    id_gen_path = os.path.join(results_dir, "fuzzy_id_generator.json")
    if os.path.exists(id_gen_path):
        print(f"  -> Fuzzy duplicate IDs saved  ->  {results_dir}/FuzzyDuplicateIds/")
        removal_workflow = TextDuplicatesRemovalWorkflow(
            input_path=input_dir,
            ids_to_remove_path=os.path.join(results_dir, "FuzzyDuplicateIds"),
            output_path=output_dir,
            input_filetype=input_filetype,
            input_id_field="_curator_dedup_id",
            ids_to_remove_duplicate_id_field="_curator_dedup_id",
            id_generator_path=id_gen_path,
            input_blocksize="512MiB",        # smaller blocksize to avoid OOM on large datasets
        )
        removal_workflow.run()
        print(f"  -> Fuzzy dedup complete  ->  {output_dir}\n")
    else:
        print("  -> No fuzzy duplicates found — copying input to output")
        shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)
        print(f"  -> Fuzzy dedup complete (no removal needed)  ->  {output_dir}\n")


# ---------------------------------------------------------------------------
# Stage 4 – Convert final parquet output to JSONL for pretraining
# ---------------------------------------------------------------------------
CURATOR_INTERNAL_COLS = {"_curator_dedup_id", _INDEX_COL}


def convert_parquet_to_jsonl(input_dir: str, output_path: str) -> int:
    """Convert parquet output to clean JSONL, dropping curator internal columns."""
    print("=" * 60)
    print("STAGE 4: Convert Parquet to JSONL")
    print("=" * 60)

    table = pq.read_table(input_dir)
    cols = [c for c in table.column_names if c not in CURATOR_INTERNAL_COLS]
    table = table.select(cols)

    # Fast column-oriented conversion (avoids slow df.iterrows())
    data = table.to_pydict()
    n = table.num_rows

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n):
            row = {k: data[k][i] for k in cols}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  -> Converted {n} records  ->  {output_path}\n")
    return n


# ---------------------------------------------------------------------------
# Main – run all stages end-to-end
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NeMo Curator Pretraining Data Pipeline: clean -> exact dedup -> fuzzy dedup",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input_data/",
        help="Path to raw input data directory (default: input_data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pipeline_output/",
        help="Base directory for all pipeline outputs (default: pipeline_output/)",
    )
    parser.add_argument(
        "--input-filetype",
        type=str,
        default="jsonl",
        choices=["jsonl", "parquet"],
        help="Input file format (default: jsonl)",
    )
    args = parser.parse_args()

    # ---- Intermediate / output directories ----
    base = args.output_dir
    cleaned_dir        = os.path.join(base, "01_cleaned")
    exact_results_dir  = os.path.join(base, "02_exact_results")
    exact_deduped_dir  = os.path.join(base, "02_exact_deduped")
    fuzzy_cache_dir    = os.path.join(base, "03_fuzzy_cache")
    fuzzy_results_dir  = os.path.join(base, "03_fuzzy_results")
    fuzzy_deduped_dir  = os.path.join(base, "03_fuzzy_deduped")
    final_jsonl_path   = os.path.join(base, "final_pretrain_data.jsonl")

    # ---- Start Ray cluster ----
    # Limit CPUs so Ray doesn't spawn hundreds of actors and OOM
    ray_client = RayClient(num_cpus=32)
    ray_client.start()

    # ---- Clean stale intermediate dirs from previous runs ----
    for d in [cleaned_dir, exact_results_dir, exact_deduped_dir,
              fuzzy_cache_dir, fuzzy_results_dir, fuzzy_deduped_dir]:
        if os.path.exists(d):
            print(f"  Removing stale directory: {d}")
            shutil.rmtree(d)
    if os.path.exists(final_jsonl_path):
        os.remove(final_jsonl_path)

    try:
        # Pre-processing: split large JSONL files into ~500 MiB chunks
        effective_input_dir = split_large_jsonl_files(args.input_dir)

        # Stage 1: Text Cleaning
        #   input_data/ (raw)  ->  01_cleaned/
        run_text_cleaning(effective_input_dir, cleaned_dir)

        # Stage 2: Exact Dedup
        #   01_cleaned/  ->  02_exact_deduped/
        run_exact_dedup(
            input_dir=cleaned_dir,
            results_dir=exact_results_dir,
            output_dir=exact_deduped_dir,
            input_filetype="jsonl",
        )

        # Fix parquet schema inconsistencies before fuzzy dedup.
        # TextDuplicatesRemovalWorkflow sometimes writes __index_level_0__
        # in some partitions but not others, causing cuDF column mismatch.
        normalize_parquet_schemas(exact_deduped_dir)

        # Stage 3: Fuzzy Dedup
        #   02_exact_deduped/  ->  03_fuzzy_deduped/  (final output)
        #   NOTE: TextDuplicatesRemovalWorkflow always outputs parquet,
        #   so Stage 3 must read parquet regardless of the original input format.
        run_fuzzy_dedup(
            input_dir=exact_deduped_dir,
            cache_dir=fuzzy_cache_dir,
            results_dir=fuzzy_results_dir,
            output_dir=fuzzy_deduped_dir,
            input_filetype="parquet",
        )

        print("=" * 60)
        print("PIPELINE COMPLETE (deduplication)")
        print("=" * 60)
        print(f"  Final deduplicated parquet  ->  {fuzzy_deduped_dir}")

        # Stage 4: Convert to JSONL (no Ray needed, but must run
        # before ray_client.stop() — stopping Ray kills the GCS
        # process which terminates the SLURM job after 60 s).
        #   03_fuzzy_deduped/ (parquet)  ->  final_pretrain_data.jsonl
        convert_parquet_to_jsonl(fuzzy_deduped_dir, final_jsonl_path)

        print("=" * 60)
        print("ALL DONE")
        print("=" * 60)
        print(f"  Pretraining-ready JSONL  ->  {final_jsonl_path}")

    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()