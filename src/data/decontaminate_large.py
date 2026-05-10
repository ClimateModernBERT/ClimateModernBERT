import argparse
import re
import unicodedata
from tqdm import tqdm
import hashlib
import gc
from collections import defaultdict


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
    normalized = " ".join(tokenize(text))
    return hashlib.md5(normalized.encode()).hexdigest()


def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def process_exact_duplicates_streaming(input_file, chunk_size=100000):
    print("\nStep 1: Removing exact duplicates (streaming mode)...")
    
    seen_hashes = set()
    duplicate_indices = set()
    total_lines = 0
    duplicates_found = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        chunk = []
        
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


def process_near_duplicates_windowed(input_file, duplicate_indices, ngram_length=15, 
                                     similarity_threshold=0.5, window_size=50000, 
                                     min_ngram_overlap=3):
    """
    Find near-duplicates using sliding window approach with n-gram Jaccard similarity.
    Uses n-gram index within window for fast candidate lookup.
    Only checks texts within a window to avoid O(n²) complexity.
    """
    print(f"\nStep 2: Finding near-duplicates (>{similarity_threshold*100:.1f}% overlap)...")
    print(f"Settings: ngram_length={ngram_length}, window_size={window_size}, min_ngram_overlap={min_ngram_overlap}")
    
    near_duplicate_indices = set()
    window_texts = []  # (line_idx, text, ngrams)
    window_ngram_index = defaultdict(set)  # ngram -> set of indices in window
    total_processed = 0
    near_dups_found = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_idx, line in enumerate(tqdm(f, desc="Processing texts for near-duplicates")):
            if line_idx in duplicate_indices:
                continue
                
            text = line.strip()
            if not text or len(text) < 10:
                continue
            
            tokens = tokenize(text)
            if len(tokens) < ngram_length:
                continue
            
            ngrams = get_ngrams(tokens, ngram_length)
            if not ngrams:
                continue
            
            # Find candidate texts that share n-grams (using index for fast lookup)
            candidate_indices = set()
            for ngram in ngrams:
                candidate_indices.update(window_ngram_index[ngram])
            
            # Check similarity only with candidates
            is_duplicate = False
            for win_idx in candidate_indices:
                if win_idx >= len(window_texts):
                    continue
                    
                win_line_idx, win_text, win_ngrams = window_texts[win_idx]
                
                # Quick filter: require minimum n-gram overlap
                overlap = len(ngrams & win_ngrams)
                if overlap < min_ngram_overlap:
                    continue
                
                # Calculate Jaccard similarity
                similarity = jaccard_similarity(ngrams, win_ngrams)
                
                if similarity > similarity_threshold:
                    # Found near-duplicate, keep the earlier one
                    near_duplicate_indices.add(line_idx)
                    near_dups_found += 1
                    is_duplicate = True
                    break
            
            # Add to window if not a duplicate
            if not is_duplicate:
                window_idx = len(window_texts)
                window_texts.append((line_idx, text, ngrams))
                total_processed += 1
                
                # Update n-gram index
                for ngram in ngrams:
                    window_ngram_index[ngram].add(window_idx)
                
                # Slide window: remove oldest entries when window is full
                if len(window_texts) >= window_size:
                    # Remove oldest entry from index
                    old_line_idx, old_text, old_ngrams = window_texts[0]
                    for ngram in old_ngrams:
                        window_ngram_index[ngram].discard(0)
                        if not window_ngram_index[ngram]:
                            del window_ngram_index[ngram]
                    
                    # Remove from window
                    window_texts.pop(0)
                    
                    # Shift all indices in the index down by 1
                    # More efficient: rebuild only when needed, but shift indices here
                    new_index = defaultdict(set)
                    for ngram, indices in window_ngram_index.items():
                        for idx in indices:
                            if idx > 0:  # Shift down by 1
                                new_index[ngram].add(idx - 1)
                    window_ngram_index = new_index
            
            # Periodic cleanup
            if line_idx % 100000 == 0:
                gc.collect()
    
    print(f"Found {near_dups_found} near-duplicates out of {total_processed} processed texts")
    return near_duplicate_indices


def self_decontaminate_large(input_file, output_file, chunk_size=100000, 
                             ngram_length=15, similarity_threshold=0.5,
                             window_size=50000, min_ngram_overlap=3, 
                             do_near_dedup=True):
    """Fast exact duplicate removal and n-gram based near-duplicate decontamination."""
    
    print(f"Self-decontaminating {input_file} (memory-efficient mode)...")
    print(f"Settings: ngram_length={ngram_length}, similarity_threshold={similarity_threshold}")
    print(f"Chunk size: {chunk_size}, Batch size: {window_size}")
    
    # Step 1: Find exact duplicates
    duplicate_indices, total_lines = process_exact_duplicates_streaming(input_file, chunk_size)
    
    # Step 2: Find near-duplicates using n-gram similarity
    if do_near_dedup:
        near_duplicate_indices = process_near_duplicates_windowed(
            input_file, duplicate_indices, ngram_length, similarity_threshold,
            window_size, min_ngram_overlap
        )
        all_duplicate_indices = duplicate_indices | near_duplicate_indices
    else:
        all_duplicate_indices = duplicate_indices
        near_duplicate_indices = set()
    
    # Step 3: Write cleaned data
    print("\nWriting cleaned data...")
    clean_count = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, \
         open(output_file, 'w', encoding='utf-8', errors='ignore') as outfile:
        for line_idx, line in enumerate(tqdm(infile, desc="Writing clean data")):
            if line_idx not in all_duplicate_indices:
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
    if do_near_dedup:
        print(f"Near-duplicates removed: {len(near_duplicate_indices)}")
    print(f"Total removed: {len(all_duplicate_indices)}")
    print(f"Clean samples saved: {clean_count}")
    print(f"Reduction: {(1 - clean_count/total_lines)*100:.1f}%")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast exact duplicate removal and n-gram based near-duplicate decontamination for large pretraining datasets."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input pretraining text file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save decontaminated text file")
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="Chunk size for exact duplicate processing (default: 100000)")
    parser.add_argument("--ngram_length", type=int, default=15,
                        help="Length of n-grams for overlap detection (default: 15)")
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                        help="Threshold for near-duplicate removal (default: 0.5 = 50%% overlap)")
    parser.add_argument("--window_size", type=int, default=50000,
                        help="Sliding window size for near-duplicate detection (default: 50000)")
    parser.add_argument("--min_ngram_overlap", type=int, default=3,
                        help="Minimum n-gram overlap required before checking similarity (default: 3)")
    parser.add_argument("--no_near_dedup", action='store_true',
                        help="Skip near-duplicate removal (only do exact duplicates)")
    
    args = parser.parse_args()
    
    self_decontaminate_large(
        input_file=args.input_file,
        output_file=args.output_file,
        chunk_size=args.chunk_size,
        ngram_length=args.ngram_length,
        similarity_threshold=args.similarity_threshold,
        window_size=args.window_size,
        min_ngram_overlap=args.min_ngram_overlap,
        do_near_dedup=not args.no_near_dedup
    )
