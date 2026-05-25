"""Stream, Filter, and Upload FineWeb Climate & Nature Data
===========================================================

This script streams the HuggingFace FineWeb-Edu dataset, filters for climate and nature
keywords, and progressively uploads matched records to a Hugging Face dataset repository
using non-blocking async uploads for efficiency.

Features:
---------
* Streams any FineWeb-Edu subset (default: sample-10BT)
* Filters with a fully-sourced keyword inventory (Sautner et al. 2023 bigrams +
  GEMET v4.2.3 climate/biosphere concepts) using strong/weak strength scoring
* Async uploads to HF Hub without blocking the pipeline
* Schema normalization to prevent dataset viewer errors
* Max samples parameter for testing before full runs
* Suitable for long-running SLURM jobs

Usage:
------
# Test run with 1000 samples
python stream_filter_upload_fineweb.py \\
    --hub-repo-id username/fineweb-climate \\
    --subset sample-10BT \\
    --max-samples 1000 \\
    --chunk-size 100

# Full production run
python stream_filter_upload_fineweb.py \\
    --hub-repo-id username/fineweb-climate \\
    --subset CC-MAIN-2024-10 \\
    --chunk-size 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:
    from datasets import Dataset, Features, IterableDataset, Sequence as DsSequence, Value, load_dataset
except ImportError as exc:
    raise ImportError(
        "The 'datasets' package is required. Install it via 'pip install datasets'."
    ) from exc

try:
    from huggingface_hub import HfApi, login
except ImportError as exc:
    raise ImportError(
        "The 'huggingface_hub' package is required. Install it via 'pip install huggingface_hub'."
    ) from exc


# ============================================================================
# KEYWORD FILTERING LOGIC
# ============================================================================

STRENGTH_STRONG = "strong"
STRENGTH_WEAK = "weak"


@dataclass(frozen=True)
class KeywordRule:
    """Compiled regex keyword with a strength tag for filtering logic."""
    
    term: str
    group: str
    strength: str
    pattern: re.Pattern[str]


KeywordSpec = Tuple[str, str, str]  # (term, group, strength)


def _compile_pattern(term: str) -> re.Pattern[str]:
    """Compile a whitespace-tolerant, word-boundary-safe regex for term.

    A '*' inside a token is treated as a stem wildcard and compiled to ``\\w*``.
    This emulates the stemmed matching used by Sautner et al. (2023) so that,
    e.g., ``'global warm*'`` matches "global warming", "global warms",
    "global warmed", and ``'reduc* emission*'`` matches "reduce emissions",
    "reducing emission", etc. Tokens without '*' are matched exactly and remain
    word-boundary safe so partial-word false positives are avoided.
    """
    tokens = term.strip().split()
    parts = []
    for tok in tokens:
        if "*" in tok:
            # Escape each literal segment, replace '*' with a letters wildcard.
            segs = tok.split("*")
            parts.append(r"\w*".join(re.escape(s) for s in segs))
        else:
            parts.append(re.escape(tok))
    joined = r"\s+".join(parts)
    pattern = rf"(?<!\w){joined}(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


def _build_rules() -> Tuple[KeywordRule, ...]:
    """Create compiled keyword rules with strength metadata."""
    
    specs: Sequence[KeywordSpec] = (

        # --- Sautner et al. (2023) 
        ('renewable energy', 'opportunity', STRENGTH_STRONG),
        ('electric vehicle*', 'opportunity', STRENGTH_STRONG),
        ('clean energy', 'opportunity', STRENGTH_STRONG),
        ('wind power', 'opportunity', STRENGTH_STRONG),
        ('wind energy', 'opportunity', STRENGTH_STRONG),
        ('solar energy', 'opportunity', STRENGTH_STRONG),
        ('plug hybrid', 'opportunity', STRENGTH_STRONG),
        ('renewable resource*', 'opportunity', STRENGTH_STRONG),
        ('solar farm*', 'opportunity', STRENGTH_STRONG),
        ('electric hybrid', 'opportunity', STRENGTH_STRONG),
        ('rooftop solar', 'opportunity', STRENGTH_STRONG),
        ('sustainable energy', 'opportunity', STRENGTH_STRONG),
        ('hybrid car*', 'opportunity', STRENGTH_STRONG),
        ('renewable electricity', 'opportunity', STRENGTH_STRONG),
        ('wave power', 'opportunity', STRENGTH_STRONG),
        ('geothermal power', 'opportunity', STRENGTH_STRONG),
        ('heat power', 'opportunity', STRENGTH_WEAK),
        ('new energy', 'opportunity', STRENGTH_WEAK),

        ('greenhouse gas*', 'regulatory', STRENGTH_STRONG),
        ('carbon emission*', 'regulatory', STRENGTH_STRONG),
        ('carbon dioxide', 'regulatory', STRENGTH_STRONG),
        ('gas emission*', 'regulatory', STRENGTH_STRONG),
        ('air pollution', 'regulatory', STRENGTH_STRONG),
        ('carbon tax', 'regulatory', STRENGTH_STRONG),
        ('carbon price*', 'regulatory', STRENGTH_STRONG),
        ('nox emission*', 'regulatory', STRENGTH_STRONG),
        ('emission* trad*', 'regulatory', STRENGTH_STRONG),
        ('dioxide emission*', 'regulatory', STRENGTH_STRONG),
        ('carbon reduction*', 'regulatory', STRENGTH_STRONG),
        ('carbon market*', 'regulatory', STRENGTH_STRONG),
        ('mercury emission*', 'regulatory', STRENGTH_STRONG),
        ('reduc* emission*', 'regulatory', STRENGTH_STRONG),
        ('reduc* carbon', 'regulatory', STRENGTH_STRONG),
        ('epa regulation*', 'regulatory', STRENGTH_STRONG),
        ('environmental standard*', 'regulatory', STRENGTH_WEAK),
        ('energy regulator*', 'regulatory', STRENGTH_WEAK),
        ('energy independence', 'regulatory', STRENGTH_WEAK),

        ('global warm*', 'physical', STRENGTH_STRONG),
        ('warm* climate', 'physical', STRENGTH_STRONG),
        ('sea level', 'physical', STRENGTH_WEAK),
        ('coastal area', 'physical', STRENGTH_WEAK),
        ('snow ice', 'physical', STRENGTH_WEAK),
        ('forest land', 'physical', STRENGTH_WEAK),
        ('natural hazard*', 'physical', STRENGTH_WEAK),
        ('sea water', 'physical', STRENGTH_WEAK),
        ('storm water', 'physical', STRENGTH_WEAK),
        ('heavy snow', 'physical', STRENGTH_WEAK),
        ('water discharge', 'physical', STRENGTH_WEAK),
        ('ice product*', 'physical', STRENGTH_WEAK),
        ('air water', 'physical', STRENGTH_WEAK),
        ('nickel metal', 'physical', STRENGTH_WEAK),

        ('air quality', 'climate', STRENGTH_WEAK),
        ('air temperature', 'climate', STRENGTH_WEAK),
        ('biomass energy', 'climate', STRENGTH_STRONG),
        ('carbon energy', 'climate', STRENGTH_STRONG),
        ('carbon neutral', 'climate', STRENGTH_STRONG),
        ('carbon sink', 'climate', STRENGTH_STRONG),
        ('clean air', 'climate', STRENGTH_WEAK),
        ('clean water', 'climate', STRENGTH_WEAK),
        ('climate change', 'climate', STRENGTH_STRONG),
        ('coastal region', 'climate', STRENGTH_WEAK),
        ('energy climate', 'climate', STRENGTH_STRONG),
        ('energy conversion', 'climate', STRENGTH_WEAK),
        ('energy efficient*', 'climate', STRENGTH_WEAK),
        ('energy environment', 'climate', STRENGTH_WEAK),
        ('environmental sustainability', 'climate', STRENGTH_STRONG),
        ('extreme weather', 'climate', STRENGTH_STRONG),
        ('flue gas', 'climate', STRENGTH_WEAK),
        ('ghg emission*', 'climate', STRENGTH_STRONG),
        ('global decarboni*', 'climate', STRENGTH_STRONG),
        ('kyoto protocol', 'climate', STRENGTH_STRONG),
        ('ozone layer', 'climate', STRENGTH_STRONG),
        ('solar thermal', 'climate', STRENGTH_STRONG),
        ('water resource*', 'climate', STRENGTH_WEAK),
        ('weather climate', 'climate', STRENGTH_STRONG),
        ('wind resource*', 'climate', STRENGTH_WEAK),

        # --- GEMET ---
        ('renewable energy source', 'climate', STRENGTH_STRONG),
        ('sustainable bioenergy production', 'climate', STRENGTH_STRONG),
        ('renewable energy directive', 'climate', STRENGTH_STRONG),
        ('net zero carbon', 'climate', STRENGTH_STRONG),
        ('ozone depletion potential', 'climate', STRENGTH_STRONG),
        ('greenhouse gas emissions', 'climate', STRENGTH_STRONG),
        ('biogas', 'climate', STRENGTH_STRONG),
        ('geothermal energy', 'climate', STRENGTH_STRONG),
        ('climate protection', 'climate', STRENGTH_STRONG),
        ('glacier', 'climate', STRENGTH_STRONG),
        ('direct greenhouse gas emissions', 'climate', STRENGTH_STRONG),
        ('atmospheric carbon dioxide', 'climate', STRENGTH_STRONG),
        ('ocean acidification', 'climate', STRENGTH_STRONG),
        ('climate regulation', 'climate', STRENGTH_STRONG),
        ('solar cell', 'climate', STRENGTH_STRONG),
        ('bioenergy production', 'climate', STRENGTH_STRONG),
        ('sea level rise', 'climate', STRENGTH_STRONG),
        ('tropospheric ozone', 'climate', STRENGTH_STRONG),
        ('ozone layer depletion', 'climate', STRENGTH_STRONG),
        ('climate change impact', 'climate', STRENGTH_STRONG),
        ('climatology', 'climate', STRENGTH_STRONG),
        ('thermo-mechanical biofuel production', 'climate', STRENGTH_STRONG),
        ('climate policy', 'climate', STRENGTH_STRONG),
        ('hydroelectric energy', 'climate', STRENGTH_STRONG),
        ('global climate', 'climate', STRENGTH_STRONG),
        ('ozone depletion', 'climate', STRENGTH_STRONG),
        ('arctic sea ice loss', 'climate', STRENGTH_STRONG),
        ('hydrometeorology', 'climate', STRENGTH_STRONG),
        ('climate effect', 'climate', STRENGTH_STRONG),
        ('solar energy technology', 'climate', STRENGTH_STRONG),
        ('bioclimatology', 'climate', STRENGTH_STRONG),
        ('climate benchmark', 'climate', STRENGTH_STRONG),
        ('anthropogenic greenhouse gas', 'climate', STRENGTH_STRONG),
        ('greenhouse gas protocol', 'climate', STRENGTH_STRONG),
        ('carbon sequestration', 'climate', STRENGTH_STRONG),
        ('greenhouse effect', 'climate', STRENGTH_STRONG),
        ('wind power station', 'climate', STRENGTH_STRONG),
        ('agricultural bioenergy production', 'climate', STRENGTH_STRONG),
        ('indirect greenhouse gas emissions', 'climate', STRENGTH_STRONG),
        ('palaeoclimatology', 'climate', STRENGTH_STRONG),
        ('global mean temperature increase', 'climate', STRENGTH_STRONG),
        ('climate-neutral economy', 'climate', STRENGTH_STRONG),
        ('hydroelectric power plant', 'climate', STRENGTH_STRONG),
        ('solar power station', 'climate', STRENGTH_STRONG),
        ('climate change adaptation', 'climate', STRENGTH_STRONG),
        ('climate change mitigation', 'climate', STRENGTH_STRONG),
        ('global temperature increase', 'climate', STRENGTH_STRONG),
        ('rising sea level', 'climate', STRENGTH_STRONG),
        ('climate alteration', 'climate', STRENGTH_STRONG),
        ('methane', 'climate', STRENGTH_STRONG),
        ('second generation biofuel production', 'climate', STRENGTH_STRONG),
        ('climate action bonds', 'climate', STRENGTH_STRONG),
        ('co2 border tax', 'climate', STRENGTH_STRONG),
        ('man-made climate change', 'climate', STRENGTH_STRONG),
        ('biofuel', 'climate', STRENGTH_STRONG),
        ('glacial retreat', 'climate', STRENGTH_STRONG),
        ('microclimatology', 'climate', STRENGTH_STRONG),
        ('algae-based biofuel production', 'climate', STRENGTH_STRONG),
        ('stratospheric ozone depletion', 'climate', STRENGTH_STRONG),
        ('permafrost ecosystem', 'climate', STRENGTH_STRONG),
        ('climate', 'climate', STRENGTH_STRONG),
        ('carbon dioxide tax', 'climate', STRENGTH_STRONG),

        # --- GEMET (weak/ambiguous) ---
        ('drought', 'climate', STRENGTH_WEAK),
        ('ecological footprint', 'climate', STRENGTH_WEAK),
        ('atmospheric circulation', 'climate', STRENGTH_WEAK),
        ('adaptive capacity', 'climate', STRENGTH_WEAK),
        ('permafrost', 'climate', STRENGTH_WEAK),
        ('water stress', 'climate', STRENGTH_WEAK),
        ('iceberg', 'climate', STRENGTH_WEAK),
        ('atmospheric ozone', 'climate', STRENGTH_WEAK),
        ('feedback loop', 'climate', STRENGTH_WEAK),
        ('ocean temperature', 'climate', STRENGTH_WEAK),
        ('climate resource', 'climate', STRENGTH_WEAK),
        ('radiative forcing', 'climate', STRENGTH_WEAK),
        ('ice sheet', 'climate', STRENGTH_WEAK),
        ('tropical climate', 'climate', STRENGTH_WEAK),
        ('environmental footprint', 'climate', STRENGTH_WEAK),
        ('meteorology', 'climate', STRENGTH_WEAK),
        ('water scarcity', 'climate', STRENGTH_WEAK),
        ('polar region', 'climate', STRENGTH_WEAK),
        ('albedo', 'climate', STRENGTH_WEAK),
        ('solar radiation', 'climate', STRENGTH_WEAK),
        ('ice sheet mass balance', 'climate', STRENGTH_WEAK),
        ('adaptation strategy', 'climate', STRENGTH_WEAK),
        ('flood forecast', 'climate', STRENGTH_WEAK),
        ('heat wave', 'climate', STRENGTH_WEAK),
        ('ice cap', 'climate', STRENGTH_WEAK),
        ('microclimate', 'climate', STRENGTH_WEAK),
        ('meteorological disaster', 'climate', STRENGTH_WEAK),
        ('ecological resilience', 'climate', STRENGTH_WEAK),

        # --- GEMET biosphere / nature concepts (strong) ---
        ('land degradation', 'nature', STRENGTH_STRONG),
        ('natural capital', 'nature', STRENGTH_STRONG),
        ('ecological corridor', 'nature', STRENGTH_STRONG),
        ('deforestation', 'nature', STRENGTH_STRONG),
        ('endangered species', 'nature', STRENGTH_STRONG),
        ('peatland', 'nature', STRENGTH_STRONG),
        ('nature conservation', 'nature', STRENGTH_STRONG),
        ('wetland', 'nature', STRENGTH_STRONG),
        ('ecosystem', 'nature', STRENGTH_STRONG),
        ('soil degradation', 'nature', STRENGTH_STRONG),
        ('biodiversity', 'nature', STRENGTH_STRONG),
        ('mangrove', 'nature', STRENGTH_STRONG),
        ('invasive species', 'nature', STRENGTH_STRONG),
        ('ecosystem services', 'nature', STRENGTH_STRONG),
        ('biodiversity loss', 'nature', STRENGTH_STRONG),
        ('protected area', 'nature', STRENGTH_STRONG),
        ('afforestation', 'nature', STRENGTH_STRONG),
        ('habitat fragmentation', 'nature', STRENGTH_STRONG),
        ('reforestation', 'nature', STRENGTH_STRONG),

        # --- GEMET biosphere / nature concepts (weak) ---
        ('biomass', 'nature', STRENGTH_WEAK),
        ('restoration', 'nature', STRENGTH_WEAK),
        ('forest management', 'nature', STRENGTH_WEAK),
        ('habitat', 'nature', STRENGTH_WEAK),
        ('land use change', 'nature', STRENGTH_WEAK),
        ('ecological', 'nature', STRENGTH_WEAK),
        ('pollinator', 'nature', STRENGTH_WEAK),
        ('wildlife', 'nature', STRENGTH_WEAK),
        ('conservation', 'nature', STRENGTH_WEAK),
    )
    
    rules = []
    for term, group, strength in specs:
        rules.append(
            KeywordRule(
                term=term,
                group=group,
                strength=strength,
                pattern=_compile_pattern(term)
            )
        )
    
    return tuple(rules)


KEYWORD_RULES: Tuple[KeywordRule, ...] = _build_rules()


@dataclass
class MatchResult:
    """Container describing keyword matches for a passage."""
    
    tags: List[str]
    matched_keywords: Dict[str, List[str]]
    strong_count: int
    weak_count: int
    total_count: int
    decision: str
    
    @property
    def has_match(self) -> bool:
        return bool(self.tags)


def normalise_text(text: str) -> str:
    """Return a lowercase, whitespace-collapsed string for regex checks."""
    return re.sub(r"\s+", " ", text.casefold())


def detect_keywords(text: str) -> MatchResult:
    """Identify climate and nature keywords using regex with strength logic."""
    
    lowered = normalise_text(text)
    matches: Dict[str, List[str]] = defaultdict(list)
    strong_hits = 0
    weak_hits = 0
    
    for rule in KEYWORD_RULES:
        if rule.pattern.search(lowered):
            matches[rule.group].append(rule.term)
            if rule.strength == STRENGTH_STRONG:
                strong_hits += 1
            else:
                weak_hits += 1
    
    total_hits = strong_hits + weak_hits
    decision = "rejected_insufficient"
    if strong_hits > 0:
        decision = "accepted_strong"
    elif total_hits >= 2:
        decision = "accepted_multi_weak"
    
    tags = sorted(matches) if decision.startswith("accepted") else []
    matches = {group: sorted(list(set(terms))) for group, terms in matches.items()} if tags else {}
    
    return MatchResult(
        tags=tags,
        matched_keywords=matches,
        strong_count=strong_hits,
        weak_count=weak_hits,
        total_count=total_hits,
        decision=decision,
    )


# ============================================================================
# STREAMING UPLOADER
# ============================================================================

class StreamingUploader:
    """
    Non-blocking uploader for streaming pipelines.
    
    Uploads shards to HF Hub without blocking the main processing loop.
    Uses run_as_future for async uploads and tracks pending uploads.
    """
    
    def __init__(self, repo_id: str, token: Optional[str] = None, temp_dir: Optional[str] = None):
        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.pending_uploads: List[Future] = []
        self.shard_counter = 0

        base_tmp = None
        if temp_dir:
            base_tmp = Path(temp_dir)
            base_tmp.mkdir(parents=True, exist_ok=True)

        self.temp_dir = tempfile.mkdtemp(prefix="fineweb_upload_", dir=str(base_tmp) if base_tmp else None)
        
        # Track nested dict keys for schema consistency across ALL shards.
        # Pre-define every keyword group so matched_keywords has a stable schema
        # even in shards where a given group never fires.
        self.nested_schema_keys: Dict[str, set] = {
            "matched_keywords": {"climate", "nature", "opportunity", "regulatory", "physical"}
        }
        
        # Create repo if needed
        self.api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        logging.info(f"✓ Repository ready: https://huggingface.co/datasets/{repo_id}")
    
    def _normalize_schema(self, data: List[dict]) -> List[dict]:
        """
        Normalize schema across all records to prevent schema mismatch between shards.
        
        This ensures fields like 'matched_keywords' have consistent structure
        (e.g., always have both 'climate' and 'nature' keys, even if empty).
        """
        # First pass: collect all nested dict keys from this batch and update global tracker
        for record in data:
            for key, value in record.items():
                if isinstance(value, dict):
                    if key not in self.nested_schema_keys:
                        self.nested_schema_keys[key] = set()
                    self.nested_schema_keys[key].update(value.keys())
        
        # Second pass: normalize all records using the global schema
        normalized = []
        for record in data:
            new_record = record.copy()
            for key, all_subkeys in self.nested_schema_keys.items():
                if key in new_record and isinstance(new_record[key], dict):
                    # Make a copy and add missing subkeys with empty lists
                    new_record[key] = new_record[key].copy()
                    for subkey in all_subkeys:
                        if subkey not in new_record[key]:
                            new_record[key][subkey] = []
                elif key not in new_record:
                    # Field doesn't exist at all, add it with empty subkeys
                    new_record[key] = {subkey: [] for subkey in all_subkeys}
            normalized.append(new_record)
        
        return normalized
    
    def _get_features(self, sample_record: dict) -> Features:
        """
        Build explicit Features schema to ensure consistent types across shards.
        This prevents empty lists from being inferred as 'null' type.
        """
        features = {}
        
        # Define schema based on known FineWeb fields + our additions
        for key, value in sample_record.items():
            if key == "tags":
                # Always a list of strings
                features[key] = DsSequence(Value("string"))
            elif key == "matched_keywords":
                # Dict with string keys -> list of strings
                features[key] = {k: DsSequence(Value("string")) for k in value.keys()}
            elif key == "match_summary":
                # Nested dict with int/string values
                features[key] = {
                    "strong": Value("int64"),
                    "weak": Value("int64"),
                    "total": Value("int64"),
                    "decision": Value("string"),
                }
            elif isinstance(value, str):
                features[key] = Value("string")
            elif isinstance(value, int):
                features[key] = Value("int64")
            elif isinstance(value, float):
                features[key] = Value("float64")
            elif isinstance(value, list):
                # Assume list of strings
                features[key] = DsSequence(Value("string"))
            elif isinstance(value, dict):
                # Generic dict handling
                features[key] = {k: Value("string") for k in value.keys()}
            else:
                features[key] = Value("string")  # fallback
        
        return Features(features)
    
    def upload_shard(self, data: List[dict]) -> Future:
        """
        Upload a chunk of data as a Parquet shard.
        
        Returns immediately and upload happens in background.
        Returns the Future object so you can check status if needed.
        """
        shard_idx = self.shard_counter
        self.shard_counter += 1
        
        # Normalize schema to prevent mismatches between shards
        normalized_data = self._normalize_schema(data)
        
        # Build explicit features schema to ensure consistent types
        features = self._get_features(normalized_data[0])
        
        # Convert to Dataset with explicit schema and save as Parquet
        ds = Dataset.from_list(normalized_data, features=features)
        shard_name = f"train-{shard_idx:05d}-of-99999.parquet"
        parquet_path = Path(self.temp_dir) / shard_name
        ds.to_parquet(parquet_path)
        
        logging.info(f"📤 Queuing shard {shard_idx:05d} ({len(data)} records)...")
        
        # Upload non-blocking
        future = self.api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=f"data/{shard_name}",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Add shard {shard_idx:05d}",
            run_as_future=True,  # 👈 Key for non-blocking!
        )
        
        self.pending_uploads.append(future)
        # Clean up completed futures
        self.pending_uploads = [f for f in self.pending_uploads if not f.done()]
        logging.info(f"   ↳ Queued (pending uploads: {len(self.pending_uploads)})")
        
        return future
    
    def wait_for_completion(self):
        """Wait for all pending uploads to complete."""
        if not self.pending_uploads:
            return
        
        logging.info(f"\n⏳ Waiting for {len(self.pending_uploads)} pending uploads...")
        for i, future in enumerate(self.pending_uploads):
            try:
                future.result()  # Block until complete
                logging.info(f"   ✓ Upload {i + 1}/{len(self.pending_uploads)} complete")
            except Exception as e:
                logging.error(f"   ✗ Upload {i + 1} failed: {e}")
        
        self.pending_uploads = []
        logging.info("✓ All uploads complete!")
    
    def cleanup(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def stream_fineweb(
    dataset_name: str,
    subset: str,
    split: str,
    token: Optional[str] = None,
    retries: int = 5,
    backoff: int = 10,
) -> IterableDataset:
    """Create a streaming dataset split with retry on network errors."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(
                dataset_name,
                name=subset,
                split=split,
                streaming=True,
                token=token,
                verification_mode="no_checks",
            )
        except Exception as exc:  # Keep broad to catch HF/network issues
            last_err = exc
            if attempt == retries:
                break
            wait_for = backoff * attempt
            logging.warning(
                "FineWeb stream load failed (attempt %s/%s): %s | retrying in %ss",
                attempt,
                retries,
                exc,
                wait_for,
            )
            time.sleep(wait_for)
    raise last_err  # type: ignore[misc]


def iter_filter_records(
    dataset: IterableDataset,
    text_column: str,
    max_samples: Optional[int] = None,
    filter_tags: Optional[List[str]] = None,
) -> Iterator[dict]:
    """Yield JSON-ready records that match the keyword criteria.
    
    Args:
        dataset: Streaming dataset to filter
        text_column: Column containing text to analyze
        max_samples: Optional limit on records to process
        filter_tags: If provided, only yield records with ALL specified tags
                    (e.g., ['climate'] for climate-only, ['nature'] for nature-only)
    """
    
    processed = 0
    matched = 0
    start_ts = time.time()
    for index, row in enumerate(dataset):
        processed += 1
        
        if max_samples is not None and processed > max_samples:
            logging.info(f"Reached max_samples limit: {max_samples}")
            break
        
        # Log progress every 5,000 records
        if processed % 5000 == 0:
            elapsed = max(time.time() - start_ts, 1e-6)
            rate = processed / elapsed
            logging.info(
                "Progress: processed=%s matched=%s rate=%.0f rec/s",
                f"{processed:,}",
                f"{matched:,}",
                rate,
            )
        
        text = row.get(text_column)
        if not isinstance(text, str):
            logging.debug(f"Skipping record {index}: missing text column '{text_column}'")
            continue
        
        match = detect_keywords(text)
        if not match.has_match:
            continue
        
        # Filter by specific tags if requested
        if filter_tags:
            if not all(tag in match.tags for tag in filter_tags):
                continue
        
        matched += 1
        # Create enriched record with match metadata
        enriched = dict(row)
        enriched["tags"] = match.tags
        enriched["matched_keywords"] = match.matched_keywords
        enriched["match_summary"] = {
            "strong": match.strong_count,
            "weak": match.weak_count,
            "total": match.total_count,
            "decision": match.decision,
        }
        
        yield enriched


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--hub-repo-id",
        required=True,
        help="Target HuggingFace dataset repository (e.g., username/fineweb-climate)"
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb-edu",
        help="Source dataset to stream (default: HuggingFaceFW/fineweb-edu)"
    )
    parser.add_argument(
        "--subset",
        default="sample-10BT",
        help="Dataset subset/config to stream (default: sample-10BT)"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to stream (default: train)"
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column containing text content to filter (default: text)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of filtered records per upload chunk (default: 500)"
    )
    parser.add_argument(
        "--temp-dir",
        default="/scratch/xxx",
        help="Directory for temporary parquet shards (default: /scratch/xxx)"
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=5,
        help="Retries for FineWeb streaming load (default: 5)"
    )
    parser.add_argument(
        "--retry-backoff",
        type=int,
        default=10,
        help="Backoff seconds multiplier between retries (default: 10)"
    )
    parser.add_argument(
        "--filter-tags",
        nargs="+",
        choices=["climate", "nature", "opportunity", "regulatory", "physical"],
        help=(
            "Only upload records carrying ALL of these tags. Groups: 'climate' & "
            "'nature' (GEMET); 'opportunity', 'regulatory', 'physical' (Sautner "
            "et al. categories). E.g. --filter-tags climate"
        )
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of records to process (for testing)"
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token (defaults to HF_TOKEN env var or hf_token.txt)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Get token
    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token:
        token_file = Path(__file__).parent / "hf_token.txt"
        if token_file.exists():
            token = token_file.read_text().strip()
    
    if token:
        login(token=token)
        logging.info("✓ Logged in to Hugging Face")
    else:
        logging.warning("⚠ No token found. Some datasets may require authentication.")
    
    # Log configuration
    logging.info("=" * 70)
    logging.info("FineWeb Climate & Nature Filter Pipeline")
    logging.info("=" * 70)
    logging.info(f"Hub Repo:     {args.hub_repo_id}")
    logging.info(f"Dataset:      {args.dataset}")
    logging.info(f"Subset:       {args.subset}")
    logging.info(f"Split:        {args.split}")
    logging.info(f"Chunk Size:   {args.chunk_size}")
    temp_dir_arg = args.temp_dir if args.temp_dir else None
    logging.info(f"Max Samples:  {args.max_samples or 'unlimited'}")
    logging.info(f"Temp Dir:     {temp_dir_arg or 'system temp'}")
    logging.info(f"Retries:      {args.retry_attempts} (backoff {args.retry_backoff}s)")
    logging.info(f"Filter Tags:  {args.filter_tags or 'all (climate + nature)'}")
    logging.info("=" * 70)
    
    # Initialize uploader
    uploader = StreamingUploader(repo_id=args.hub_repo_id, token=token, temp_dir=temp_dir_arg)
    
    # Stream and filter
    logging.info("\n🚀 Starting streaming pipeline...")
    start_time = time.time()
    
    dataset = stream_fineweb(
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        token=token,
        retries=args.retry_attempts,
        backoff=args.retry_backoff,
    )
    
    filtered_iter = iter_filter_records(
        dataset=dataset,
        text_column=args.text_column,
        max_samples=args.max_samples,
        filter_tags=args.filter_tags
    )
    
    # Process in chunks
    buffer = []
    total_matched = 0
    
    for record in filtered_iter:
        buffer.append(record)
        total_matched += 1
        
        # Upload when buffer reaches chunk size
        if len(buffer) >= args.chunk_size:
            uploader.upload_shard(buffer)
            buffer = []
            logging.info(f"   → Total matched records: {total_matched:,}")
    
    # Upload any remaining data
    if buffer:
        uploader.upload_shard(buffer)
    
    # Wait for all uploads to complete
    uploader.wait_for_completion()
    uploader.cleanup()
    
    elapsed = time.time() - start_time
    logging.info("\n" + "=" * 70)
    logging.info("🎉 Pipeline Complete!")
    logging.info(f"   Matched records: {total_matched:,}")
    logging.info(f"   Time elapsed:    {elapsed:.2f}s")
    logging.info(f"   View dataset:    https://huggingface.co/datasets/{args.hub_repo_id}")
    if total_matched == 0:
        logging.warning("No matches were found. Check subset/text column/token availability.")
    logging.info("=" * 70)
    logging.info(f"\n💡 Load with: load_dataset('{args.hub_repo_id}')")


if __name__ == "__main__":
    main()
