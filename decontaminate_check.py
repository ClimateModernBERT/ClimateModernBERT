
import argparse
import difflib
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import hashlib
import numpy as np


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
    
    # Calculate bidirectional similarity
    ratio1 = len(match) / len(tokens1) if len(tokens1) > 0 else 0
    ratio2 = len(match) / len(tokens2) if len(tokens2) > 0 else 0
    
    return max(ratio1, ratio2)


def analyze_contamination(input_file, ngram_lengths=[10, 15], similarity_thresholds=[0.3, 0.5, 0.7, 0.9], 
                         sample_size=None, detailed_report=False):
    """
    Analyze contamination in a dataset for different n-gram lengths and thresholds.
    
    Args:
        input_file: Input text file (one sample per line)
        ngram_lengths: List of n-gram lengths to test
        similarity_thresholds: List of similarity thresholds to test
        sample_size: Optional sample size for faster analysis (None = use all)
        detailed_report: Whether to save detailed contamination pairs
    """
    
    print(f"Analyzing contamination in {input_file}...")
    print(f"N-gram lengths to test: {ngram_lengths}")
    print(f"Similarity thresholds to test: {similarity_thresholds}")
    print("="*60)
    
    # Load texts
    print("\nLoading texts...")
    texts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading"):
            text = line.strip()
            if text and len(text) >= 10:  # Skip very short texts
                texts.append(text)
    
    # Sample if requested
    if sample_size and sample_size < len(texts):
        import random
        random.seed(42)  # For reproducibility
        texts = random.sample(texts, sample_size)
        print(f"Using sample of {sample_size} texts for analysis")
    
    total_texts = len(texts)
    print(f"Total texts to analyze: {total_texts}")
    
    # Find exact duplicates first
    print("\n" + "="*60)
    print("EXACT DUPLICATE ANALYSIS")
    print("="*60)
    seen_hashes = {}
    exact_duplicates = 0
    exact_duplicate_groups = defaultdict(list)
    
    for i, text in enumerate(tqdm(texts, desc="Finding exact duplicates")):
        text_hash = get_text_hash(text)
        if text_hash in seen_hashes:
            exact_duplicates += 1
            exact_duplicate_groups[text_hash].append(i)
        else:
            seen_hashes[text_hash] = i
            exact_duplicate_groups[text_hash].append(i)
    
    # Count groups with duplicates
    duplicate_groups = sum(1 for group in exact_duplicate_groups.values() if len(group) > 1)
    max_duplicates = max(len(group) for group in exact_duplicate_groups.values())
    
    print(f"\nExact Duplicate Statistics:")
    print(f"  Total texts: {total_texts}")
    print(f"  Unique texts: {total_texts - exact_duplicates}")
    print(f"  Exact duplicates: {exact_duplicates}")
    print(f"  Duplicate rate: {exact_duplicates/total_texts*100:.2f}%")
    print(f"  Unique duplicate groups: {duplicate_groups}")
    print(f"  Max duplicates of same text: {max_duplicates}")
    
    # Analyze near-duplicates for each n-gram length
    results = {}
    
    for ngram_length in ngram_lengths:
        print("\n" + "="*60)
        print(f"ANALYZING {ngram_length}-GRAM OVERLAPS")
        print("="*60)
        
        # Build n-gram index
        print(f"\nBuilding {ngram_length}-gram index...")
        ngram_to_texts = defaultdict(set)
        text_ngrams = {}
        total_ngrams = 0
        texts_with_ngrams = 0
        
        for i, text in enumerate(tqdm(texts, desc=f"Building {ngram_length}-grams")):
            tokens = tokenize(text)
            if len(tokens) >= ngram_length:
                ngrams = get_ngrams(tokens, ngram_length)
                if ngrams:
                    text_ngrams[i] = ngrams
                    texts_with_ngrams += 1
                    total_ngrams += len(ngrams)
                    for ngram in ngrams:
                        ngram_to_texts[ngram].add(i)
        
        unique_ngrams = len(ngram_to_texts)
        avg_ngrams_per_text = total_ngrams / texts_with_ngrams if texts_with_ngrams > 0 else 0
        
        print(f"\n{ngram_length}-gram Statistics:")
        print(f"  Texts with enough tokens: {texts_with_ngrams}/{total_texts}")
        print(f"  Total {ngram_length}-grams: {total_ngrams:,}")
        print(f"  Unique {ngram_length}-grams: {unique_ngrams:,}")
        print(f"  Avg {ngram_length}-grams per text: {avg_ngrams_per_text:.1f}")
        print(f"  {ngram_length}-gram reuse rate: {(1 - unique_ngrams/total_ngrams)*100:.2f}%")
        
        # Find texts sharing n-grams
        print(f"\nFinding texts with {ngram_length}-gram overlaps...")
        overlap_pairs = set()
        texts_with_overlaps = set()
        
        for i in tqdm(range(len(texts)), desc="Finding overlaps"):
            if i not in text_ngrams:
                continue
            
            # Find other texts sharing n-grams
            candidates = set()
            for ngram in text_ngrams[i]:
                candidates.update(ngram_to_texts[ngram])
            candidates.discard(i)
            
            if candidates:
                texts_with_overlaps.add(i)
                for j in candidates:
                    if j > i:  # Only count each pair once
                        overlap_pairs.add((i, j))
        
        print(f"\n{ngram_length}-gram Overlap Statistics:")
        print(f"  Texts with ANY {ngram_length}-gram overlap: {len(texts_with_overlaps)}/{total_texts} ({len(texts_with_overlaps)/total_texts*100:.2f}%)")
        print(f"  Total overlapping pairs: {len(overlap_pairs):,}")
        
        # Calculate similarity for different thresholds
        print(f"\nCalculating similarity ratios for {len(overlap_pairs):,} pairs...")
        threshold_stats = {thresh: 0 for thresh in similarity_thresholds}
        similarity_distribution = []
        
        # Sample pairs if too many (for performance)
        pairs_to_check = list(overlap_pairs)
        if len(pairs_to_check) > 10000:
            import random
            random.seed(42)
            pairs_to_check = random.sample(pairs_to_check, 10000)
            print(f"  (Sampling 10,000 pairs for similarity calculation)")
        
        contaminated_pairs_by_threshold = defaultdict(list)
        
        for i, j in tqdm(pairs_to_check, desc="Calculating similarities"):
            similarity = calculate_similarity(texts[i], texts[j])
            similarity_distribution.append(similarity)
            
            for thresh in similarity_thresholds:
                if similarity > thresh:
                    threshold_stats[thresh] += 1
                    if detailed_report:
                        contaminated_pairs_by_threshold[thresh].append({
                            'pair': (i, j),
                            'similarity': similarity,
                            'text1_preview': texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                            'text2_preview': texts[j][:100] + "..." if len(texts[j]) > 100 else texts[j]
                        })
        
        # Calculate statistics
        if similarity_distribution:
            similarity_distribution = np.array(similarity_distribution)
            
            print(f"\n{ngram_length}-gram Similarity Distribution:")
            print(f"  Min similarity: {similarity_distribution.min():.3f}")
            print(f"  Max similarity: {similarity_distribution.max():.3f}")
            print(f"  Mean similarity: {similarity_distribution.mean():.3f}")
            print(f"  Median similarity: {np.median(similarity_distribution):.3f}")
            print(f"  Std deviation: {similarity_distribution.std():.3f}")
            
            # Percentiles
            percentiles = [25, 50, 75, 90, 95, 99]
            print(f"\n  Percentiles:")
            for p in percentiles:
                print(f"    {p}th percentile: {np.percentile(similarity_distribution, p):.3f}")
        
        print(f"\n{ngram_length}-gram Contamination by Threshold:")
        for thresh in sorted(similarity_thresholds):
            contaminated = threshold_stats[thresh]
            if len(pairs_to_check) < len(overlap_pairs):
                # Estimate for full dataset
                estimated = int(contaminated * len(overlap_pairs) / len(pairs_to_check))
                print(f"  Threshold {thresh:.1f}: ~{estimated:,} pairs (estimated)")
            else:
                print(f"  Threshold {thresh:.1f}: {contaminated:,} pairs")
            
            # Estimate texts affected (rough approximation)
            estimated_texts = min(contaminated * 2, total_texts)  # Each pair involves 2 texts
            print(f"    → Approximately {estimated_texts:,}/{total_texts} texts affected ({estimated_texts/total_texts*100:.2f}%)")
        
        # Store results
        results[ngram_length] = {
            'total_texts': total_texts,
            'texts_with_overlaps': len(texts_with_overlaps),
            'overlap_pairs': len(overlap_pairs),
            'threshold_stats': threshold_stats,
            'similarity_distribution': similarity_distribution if similarity_distribution else None,
            'contaminated_pairs': contaminated_pairs_by_threshold if detailed_report else None
        }
    
    # Generate summary report
    print("\n" + "="*60)
    print("CONTAMINATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nDataset: {input_file}")
    print(f"Total texts analyzed: {total_texts}")
    
    print("\nRecommended Decontamination Impact:")
    for ngram_length in ngram_lengths:
        if ngram_length in results:
            result = results[ngram_length]
            for thresh in [0.5]:  # Focus on 0.5 threshold as per original requirement
                if thresh in result['threshold_stats']:
                    pairs = result['threshold_stats'][thresh]
                    if len(pairs_to_check) < result['overlap_pairs']:
                        pairs = int(pairs * result['overlap_pairs'] / len(pairs_to_check))
                    estimated_texts = min(pairs * 2, total_texts)
                    print(f"\n{ngram_length}-gram with {thresh:.1f} threshold:")
                    print(f"  Would remove approximately {estimated_texts:,} texts ({estimated_texts/total_texts*100:.1f}%)")
    
    # Save detailed report if requested
    if detailed_report:
        report_file = Path(input_file).stem + "_contamination_report.txt"
        with open(report_file, 'w') as f:
            f.write("DETAILED CONTAMINATION ANALYSIS REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Dataset: {input_file}\n")
            f.write(f"Total texts: {total_texts}\n\n")
            
            for ngram_length in ngram_lengths:
                if ngram_length in results:
                    f.write(f"\n{ngram_length}-GRAM ANALYSIS\n")
                    f.write("-"*40 + "\n")
                    
                    result = results[ngram_length]
                    for thresh in sorted(similarity_thresholds):
                        if thresh in result['threshold_stats'] and result['contaminated_pairs']:
                            pairs = result['contaminated_pairs'][thresh]
                            if pairs:
                                f.write(f"\nThreshold {thresh:.1f}: {len(pairs)} contaminated pairs\n")
                                for i, pair_info in enumerate(pairs[:10]):  # Show first 10
                                    f.write(f"\nPair {i+1}: Similarity = {pair_info['similarity']:.3f}\n")
                                    f.write(f"  Text 1: {pair_info['text1_preview']}\n")
                                    f.write(f"  Text 2: {pair_info['text2_preview']}\n")
                                if len(pairs) > 10:
                                    f.write(f"  ... and {len(pairs)-10} more pairs\n")
        
        print(f"\nDetailed report saved to: {report_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze contamination in a pretraining dataset without modifying it."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input text file to analyze")
    parser.add_argument("--ngram_lengths", type=int, nargs='+', default=[10, 15],
                        help="N-gram lengths to test (default: 10 15)")
    parser.add_argument("--thresholds", type=float, nargs='+', 
                        default=[0.3, 0.5, 0.7, 0.9],
                        help="Similarity thresholds to test (default: 0.3 0.5 0.7 0.9)")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Sample size for faster analysis (default: use all texts)")
    parser.add_argument("--detailed_report", action='store_true',
                        help="Save detailed contamination pairs to file")
    
    args = parser.parse_args()
    
    analyze_contamination(
        input_file=args.input_file,
        ngram_lengths=args.ngram_lengths,
        similarity_thresholds=args.thresholds,
        sample_size=args.sample_size,
        detailed_report=args.detailed_report
    )