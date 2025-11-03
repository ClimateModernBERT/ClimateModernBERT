#!/usr/bin/env python3
"""
Memory-efficient self-decontamination for large pretraining datasets.
Processes data in chunks and uses disk-based storage for n-gram index when needed.
"""

import argparse
import difflib
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import hashlib
import pickle
import os
import gc
import tempfile
import shutil


def tokenize(text):
    """Normalize text by removing diacritics and tokenize."""
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    tokens = re.findall(r"\w+", text.lower())
    return tokens


def get_ngrams(tokens, n):
    """Generate n-grams from tokens."""
    if len(tokens) < n:
        return set()
    return set(zip(*[tokens[i:] for i in range(n)]))


def get_text_hash(text):
    """Get hash of normalized text for exact duplicate detection."""
    normalized = " ".join(tokenize(text))
    return hashlib.md5(normalized.encode()).hexdigest()


def diff_strings(string1, string2):
    """Find matching parts between two strings."""
    matcher = difflib.SequenceMatcher(None, string1.lower(), string2.lower(), autojunk=False)
    matching_blocks = matcher.get_matching_blocks()
    matches = []
    for block in matching_blocks:
        start_a, start_b, length = block
        if length > 5:
            match = string1[start_a:start_a + length]
            matches.append(match)
    return matches


def calculate_similarity(text1, text2):
    """Calculate similarity ratio between two texts."""
    tokens1 = " ".join(tokenize(text1))
    tokens2 = " ".join(tokenize(text2))
    
    if not tokens1 or not tokens2:
        return 0.0
    
    matching_parts = diff_strings(tokens1, tokens2)
    match = " ".join("".join(matching_parts).split())
    
    ratio1 = len(match) / len(tokens1) if len(tokens1) > 0 else 0
    ratio2 = len(match) / len(tokens2) if len(tokens2) > 0 else 0
    
    return max(ratio1, ratio2)


class DiskBasedNgramIndex:
    """Disk-based n-gram index for handling large datasets."""
    
    def __init__(self, temp_dir, max_memory_items=1000000):
        self.temp_dir = temp_dir
        self.max_memory_items = max_memory_items
        self.memory_index = {}
        self.disk_indices = []
        self.current_items = 0
        
    def add(self, ngram, text_id):
        """Add an n-gram to text_id mapping."""
        if self.current_items >= self.max_memory_items:
            self._flush_to_disk()
        
        if ngram not in self.memory_index:
            self.memory_index[ngram] = set()
        self.memory_index[ngram].add(text_id)
        self.current_items += 1
    
    def _flush_to_disk(self):
        """Flush current memory index to disk."""
        if self.memory_index:
            disk_file = os.path.join(self.temp_dir, f"index_{len(self.disk_indices)}.pkl")
            with open(disk_file, 'wb') as f:
                pickle.dump(self.memory_index, f)
            self.disk_indices.append(disk_file)
            self.memory_index = {}
            self.current_items = 0
            gc.collect()
    
    def get_texts_for_ngram(self, ngram):
        """Get all text IDs that contain this n-gram."""
        texts = set()
        
        # Check memory index
        if ngram in self.memory_index:
            texts.update(self.memory_index[ngram])
        
        # Check disk indices
        for disk_file in self.disk_indices:
            with open(disk_file, 'rb') as f:
                disk_index = pickle.load(f)
                if ngram in disk_index:
                    texts.update(disk_index[ngram])
        
        return texts
    
    def finalize(self):
        """Finalize the index (flush remaining items)."""
        self._flush_to_disk()


def process_exact_duplicates_streaming(input_file, chunk_size=100000):
    """Process exact duplicates in a streaming fashion."""
    print("\nStep 1: Removing exact duplicates (streaming mode)...")
    
    seen_hashes = set()
    duplicate_indices = set()
    total_lines = 0
    duplicates_found = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        chunk = []
        chunk_start_idx = 0
        
        for line_idx, line in enumerate(tqdm(f, desc="Finding exact duplicates")):
            text = line.strip()
            if not text or len(text) < 10:
                duplicate_indices.add(line_idx)
                continue
            
            chunk.append((line_idx, text))
            total_lines += 1
            
            # Process chunk when it reaches chunk_size
            if len(chunk) >= chunk_size:
                for idx, text in chunk:
                    text_hash = get_text_hash(text)
                    if text_hash in seen_hashes:
                        duplicate_indices.add(idx)
                        duplicates_found += 1
                    else:
                        seen_hashes.add(text_hash)
                
                chunk = []
                gc.collect()
        
        # Process remaining chunk
        for idx, text in chunk:
            text_hash = get_text_hash(text)
            if text_hash in seen_hashes:
                duplicate_indices.add(idx)
                duplicates_found += 1
            else:
                seen_hashes.add(text_hash)
    
    print(f"Found {duplicates_found} exact duplicates out of {total_lines} texts")
    return duplicate_indices, total_lines


def build_ngram_index_chunked(input_file, duplicate_indices, ngram_length, chunk_size=10000, temp_dir=None):
    """Build n-gram index in chunks to manage memory."""
    print(f"\nBuilding {ngram_length}-gram index (chunked processing)...")
    
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="ngram_index_")
    
    # Use disk-based index for very large datasets
    ngram_index = DiskBasedNgramIndex(temp_dir)
    text_ngrams_file = os.path.join(temp_dir, "text_ngrams.pkl")
    
    text_ngrams = {}
    processed_texts = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        chunk_buffer = []
        
        for line_idx, line in enumerate(tqdm(f, desc=f"Building {ngram_length}-grams")):
            if line_idx in duplicate_indices:
                continue
            
            text = line.strip()
            if not text:
                continue
            
            chunk_buffer.append((line_idx, text))
            
            # Process chunk
            if len(chunk_buffer) >= chunk_size:
                for idx, text in chunk_buffer:
                    tokens = tokenize(text)
                    if len(tokens) >= ngram_length:
                        ngrams = get_ngrams(tokens, ngram_length)
                        if ngrams:
                            text_ngrams[idx] = ngrams
                            for ngram in ngrams:
                                ngram_index.add(ngram, idx)
                            processed_texts += 1
                
                chunk_buffer = []
                
                # Periodically save text_ngrams to disk
                if processed_texts % 50000 == 0:
                    with open(text_ngrams_file + f".{processed_texts}", 'wb') as f:
                        pickle.dump(text_ngrams, f)
                    text_ngrams = {}
                    gc.collect()
        
        # Process remaining buffer
        for idx, text in chunk_buffer:
            tokens = tokenize(text)
            if len(tokens) >= ngram_length:
                ngrams = get_ngrams(tokens, ngram_length)
                if ngrams:
                    text_ngrams[idx] = ngrams
                    for ngram in ngrams:
                        ngram_index.add(ngram, idx)
                    processed_texts += 1
    
    # Save final text_ngrams
    with open(text_ngrams_file + f".final", 'wb') as f:
        pickle.dump(text_ngrams, f)
    
    ngram_index.finalize()
    
    print(f"Processed {processed_texts} texts with {ngram_length}-grams")
    return ngram_index, text_ngrams_file, temp_dir


def find_near_duplicates_batched(input_file, duplicate_indices, ngram_index, text_ngrams_file, 
                                ngram_length, similarity_threshold, temp_dir, batch_size=1000):
    """Find near-duplicates in batches to manage memory."""
    print(f"\nFinding near-duplicates (>{similarity_threshold*100}% overlap)...")
    
    near_duplicate_indices = set()
    checked_pairs = set()
    pairs_checked = 0
    near_duplicates_found = 0
    
    # Process texts in batches
    batch_texts = {}
    batch_indices = []
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_idx, line in enumerate(tqdm(f, desc="Processing texts for near-duplicates")):
            if line_idx in duplicate_indices:
                continue
            
            text = line.strip()
            if not text:
                continue
            
            batch_texts[line_idx] = text
            batch_indices.append(line_idx)
            
            # Process batch
            if len(batch_indices) >= batch_size:
                # Load text ngrams for this batch
                text_ngrams = {}
                for ngram_file in Path(temp_dir).glob("text_ngrams.pkl*"):
                    with open(ngram_file, 'rb') as f:
                        stored_ngrams = pickle.load(f)
                        for idx in batch_indices:
                            if idx in stored_ngrams:
                                text_ngrams[idx] = stored_ngrams[idx]
                
                # Check for near-duplicates within batch
                for i, idx1 in enumerate(batch_indices):
                    if idx1 in near_duplicate_indices or idx1 not in text_ngrams:
                        continue
                    
                    # Find candidate texts that share n-grams
                    candidates = set()
                    for ngram in list(text_ngrams[idx1])[:100]:  # Limit ngrams checked for memory
                        candidate_texts = ngram_index.get_texts_for_ngram(ngram)
                        candidates.update(candidate_texts)
                    
                    candidates.discard(idx1)
                    
                    # Check similarity with candidates
                    for idx2 in candidates:
                        if idx2 <= idx1 or idx2 in near_duplicate_indices:
                            continue
                        
                        pair = (idx1, idx2)
                        if pair in checked_pairs:
                            continue
                        
                        checked_pairs.add(pair)
                        pairs_checked += 1
                        
                        # Load text2 if not in current batch
                        if idx2 not in batch_texts:
                            continue  # Skip pairs across batches for memory efficiency
                        
                        # Calculate similarity
                        similarity = calculate_similarity(batch_texts[idx1], batch_texts[idx2])
                        
                        if similarity > similarity_threshold:
                            near_duplicate_indices.add(idx2)
                            near_duplicates_found += 1
                        
                        # Memory management
                        if pairs_checked % 10000 == 0:
                            gc.collect()
                
                # Clear batch
                batch_texts = {}
                batch_indices = []
                gc.collect()
    
    print(f"Found {near_duplicates_found} near-duplicates")
    return near_duplicate_indices


def self_decontaminate_large(input_file, output_file, ngram_length=15, similarity_threshold=0.5,
                            chunk_size=10000, batch_size=1000, skip_near_dedup=False):
    """
    Memory-efficient self-decontamination for large datasets.
    """
    
    print(f"Self-decontaminating {input_file} (memory-efficient mode)...")
    print(f"Settings: ngram_length={ngram_length}, similarity_threshold={similarity_threshold}")
    print(f"Chunk size: {chunk_size}, Batch size: {batch_size}")
    
    temp_dir = tempfile.mkdtemp(prefix="decontam_")
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Step 1: Find exact duplicates
        duplicate_indices, total_lines = process_exact_duplicates_streaming(input_file, chunk_size)
        all_removed_indices = duplicate_indices.copy()
        
        # Step 2: Find near-duplicates (optional)
        if not skip_near_dedup:
            ngram_index, text_ngrams_file, _ = build_ngram_index_chunked(
                input_file, duplicate_indices, ngram_length, chunk_size, temp_dir
            )
            
            near_duplicate_indices = find_near_duplicates_batched(
                input_file, duplicate_indices, ngram_index, text_ngrams_file,
                ngram_length, similarity_threshold, temp_dir, batch_size
            )
            
            all_removed_indices.update(near_duplicate_indices)
            print(f"Total near-duplicates found: {len(near_duplicate_indices)}")
        
        # Step 3: Write cleaned data
        print("\nWriting cleaned data...")
        clean_count = 0
        
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, \
             open(output_file, 'w', encoding='utf-8', errors='ignore') as outfile:
            for line_idx, line in enumerate(tqdm(infile, desc="Writing clean data")):
                if line_idx not in all_removed_indices:
                    text = line.strip()
                    if text and len(text) >= 10:
                        outfile.write(line)
                        clean_count += 1
        
        # Print summary
        print("\n" + "="*50)
        print("DECONTAMINATION COMPLETE")
        print("="*50)
        print(f"Original samples: {total_lines}")
        print(f"Exact duplicates removed: {len(duplicate_indices)}")
        if not skip_near_dedup:
            print(f"Near-duplicates removed: {len(near_duplicate_indices)}")
        print(f"Total removed: {len(all_removed_indices)}")
        print(f"Clean samples saved: {clean_count}")
        print(f"Reduction: {(1 - clean_count/total_lines)*100:.1f}%")
        print(f"Output saved to: {output_file}")
        
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temp directory: {temp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Memory-efficient self-decontamination for large pretraining datasets."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input pretraining text file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save decontaminated text file")
    parser.add_argument("--ngram_length", type=int, default=15,
                        help="Length of n-grams for overlap detection (default: 15)")
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                        help="Threshold for near-duplicate removal (default: 0.5)")
    parser.add_argument("--chunk_size", type=int, default=10000,
                        help="Chunk size for processing (default: 10000)")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for similarity checking (default: 1000)")
    parser.add_argument("--skip_near_dedup", action='store_true',
                        help="Only remove exact duplicates (skip near-duplicate detection)")
    
    args = parser.parse_args()
    
    self_decontaminate_large(
        input_file=args.input_file,
        output_file=args.output_file,
        ngram_length=args.ngram_length,
        similarity_threshold=args.similarity_threshold,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        skip_near_dedup=args.skip_near_dedup
    )
