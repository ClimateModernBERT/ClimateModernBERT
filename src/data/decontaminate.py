
import argparse
import difflib
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import hashlib


def tokenize(text):
    """Normalize text by removing diacritics and tokenize."""
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    tokens = re.findall(r"\w+", text.lower())
    return tokens


def get_ngrams(tokens, n):
    """Generate n-grams from tokens."""
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
    
    # Handle empty texts
    if not tokens1 or not tokens2:
        return 0.0
    
    matching_parts = diff_strings(tokens1, tokens2)
    match = " ".join("".join(matching_parts).split())
    
    # Calculate bidirectional similarity (check contamination both ways)
    ratio1 = len(match) / len(tokens1) if len(tokens1) > 0 else 0
    ratio2 = len(match) / len(tokens2) if len(tokens2) > 0 else 0
    
    # Return the maximum ratio (if either text contains most of the other)
    return max(ratio1, ratio2)


def self_decontaminate(input_file, output_file, ngram_length=15, similarity_threshold=0.5, 
                       exact_dedup=True, near_dedup=True, min_text_length=10):
    
    print(f"Self-decontaminating {input_file}...")
    print(f"Settings: ngram_length={ngram_length}, similarity_threshold={similarity_threshold}")
    print(f"Exact dedup: {exact_dedup}, Near dedup: {near_dedup}")
    
    # Load all texts
    print("Loading texts...")
    texts = []
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading")):
            text = line.strip()
            if text and len(text) >= min_text_length:
                texts.append((line_num, text))
    
    print(f"Loaded {len(texts)} texts (filtered out very short texts < {min_text_length} chars)")
    
    # Track which texts to keep
    keep_indices = set(range(len(texts)))
    stats = {
        "exact_duplicates": 0,
        "near_duplicates": 0,
        "short_texts": 0
    }
    
    # Step 1: Remove exact duplicates
    if exact_dedup:
        print("\nStep 1: Removing exact duplicates...")
        seen_hashes = {}
        for i, (line_num, text) in enumerate(tqdm(texts, desc="Exact dedup")):
            text_hash = get_text_hash(text)
            if text_hash in seen_hashes:
                # This is a duplicate, remove it
                if i in keep_indices:
                    keep_indices.remove(i)
                    stats["exact_duplicates"] += 1
            else:
                seen_hashes[text_hash] = i
        print(f"Removed {stats['exact_duplicates']} exact duplicates")
    
    # Step 2: Remove near-duplicates using n-gram overlap
    if near_dedup:
        print(f"\nStep 2: Removing near-duplicates (>{similarity_threshold*100}% overlap)...")
        
        # Build n-gram index for remaining texts
        print("Building n-gram index...")
        ngram_to_texts = defaultdict(set)
        text_ngrams = {}
        
        for i in tqdm(keep_indices, desc="Building n-grams"):
            _, text = texts[i]
            tokens = tokenize(text)
            ngrams = get_ngrams(tokens, ngram_length)
            text_ngrams[i] = ngrams
            for ngram in ngrams:
                ngram_to_texts[ngram].add(i)
        
        # Find near-duplicates
        print("Finding near-duplicates...")
        checked_pairs = set()
        to_remove = set()
        
        for i in tqdm(keep_indices, desc="Checking similarity"):
            if i in to_remove:
                continue
                
            # Find texts that share n-grams with this text
            candidates = set()
            for ngram in text_ngrams.get(i, []):
                candidates.update(ngram_to_texts[ngram])
            candidates.discard(i)  # Don't compare with itself
            
            # Check similarity with candidates
            for j in candidates:
                if j in to_remove or j <= i:  # Only check each pair once
                    continue
                    
                pair = (min(i, j), max(i, j))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                
                # Calculate similarity
                _, text1 = texts[i]
                _, text2 = texts[j]
                similarity = calculate_similarity(text1, text2)
                
                if similarity > similarity_threshold:
                    # Remove the later text (keep the first occurrence)
                    to_remove.add(j)
                    stats["near_duplicates"] += 1
        
        # Remove near-duplicates
        keep_indices -= to_remove
        print(f"Removed {stats['near_duplicates']} near-duplicates")
    
    # Step 3: Save cleaned data
    print("\nSaving cleaned data...")
    clean_count = 0
    
    # Sort kept indices to maintain original order
    kept_indices_sorted = sorted(keep_indices)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in tqdm(kept_indices_sorted, desc="Writing clean data"):
            _, text = texts[i]
            f.write(text + '\n')
            clean_count += 1
    
    # Print summary
    print("\n" + "="*50)
    print("DECONTAMINATION COMPLETE")
    print("="*50)
    print(f"Original samples: {len(texts)}")
    print(f"Exact duplicates removed: {stats['exact_duplicates']}")
    print(f"Near-duplicates removed: {stats['near_duplicates']}")
    print(f"Total removed: {stats['exact_duplicates'] + stats['near_duplicates']}")
    print(f"Clean samples saved: {clean_count}")
    print(f"Reduction: {(1 - clean_count/len(texts))*100:.1f}%")
    print(f"Output saved to: {output_file}")
    
    # Optionally save a detailed report
    report_file = output_file.replace('.txt', '_report.txt')
    with open(report_file, 'w') as f:
        f.write("Self-Decontamination Report\n")
        f.write("="*50 + "\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"N-gram length: {ngram_length}\n")
        f.write(f"Similarity threshold: {similarity_threshold}\n")
        f.write(f"\nStatistics:\n")
        f.write(f"Original samples: {len(texts)}\n")
        f.write(f"Exact duplicates: {stats['exact_duplicates']}\n")
        f.write(f"Near-duplicates: {stats['near_duplicates']}\n")
        f.write(f"Clean samples: {clean_count}\n")
        f.write(f"Reduction: {(1 - clean_count/len(texts))*100:.1f}%\n")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-decontaminate a pretraining dataset by removing duplicates and near-duplicates."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input pretraining text file (one sample per line)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the decontaminated text file")
    parser.add_argument("--ngram_length", type=int, default=15,
                        help="Length of n-grams for overlap detection (default: 15)")
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                        help="Threshold for near-duplicate removal (default: 0.5 = 50%% overlap)")
    parser.add_argument("--no_exact_dedup", action='store_true',
                        help="Skip exact duplicate removal")
    parser.add_argument("--no_near_dedup", action='store_true',
                        help="Skip near-duplicate removal")
    parser.add_argument("--min_text_length", type=int, default=10,
                        help="Minimum text length to keep (default: 10 characters)")
    
    args = parser.parse_args()
    
    self_decontaminate(
        input_file=args.input_file,
        output_file=args.output_file,
        ngram_length=args.ngram_length,
        similarity_threshold=args.similarity_threshold,
        exact_dedup=not args.no_exact_dedup,
        near_dedup=not args.no_near_dedup,
        min_text_length=args.min_text_length
    )