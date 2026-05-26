"""
Combine multiple data sources into a single MDS dataset for ModernBERT pretraining.
"""
import json
import csv
from pathlib import Path
from typing import Iterator, Dict, Any
from streaming import MDSWriter
from datasets import load_dataset
from tqdm import tqdm


def read_jsonl(filepath: str) -> Iterator[Dict[str, Any]]:
    """Read JSONL file, yielding one dict per line."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json(filepath: str, text_key: str = 'text') -> Iterator[Dict[str, Any]]:
    """Read JSON file (assumes list of objects or object with list value)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list of dicts and dict with a list
    if isinstance(data, list):
        for item in data:
            yield item
    elif isinstance(data, dict):
        # Try common patterns like {"data": [...]} or {"documents": [...]}
        for key in ['data', 'documents', 'texts', 'samples']:
            if key in data and isinstance(data[key], list):
                for item in data[key]:
                    yield item
                return
        # If it's a single document
        yield data


def read_csv(filepath: str, text_column: str = 'text', delimiter: str = ',') -> Iterator[Dict[str, Any]]:
    """Read CSV/TSV file."""
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            yield row


def read_txt(filepath: str, doc_separator: str = '\n\n') -> Iterator[Dict[str, Any]]:
    """Read plain text file, splitting by separator (default: double newline)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into documents
    documents = content.split(doc_separator)
    for doc in documents:
        doc = doc.strip()
        if doc:
            yield {'text': doc}


def read_huggingface(
    dataset_name: str,
    split: str = 'train',
    text_column: str = 'text',
    subset: str = None,
    streaming: bool = True
) -> Iterator[Dict[str, Any]]:
    """Read from HuggingFace datasets."""
    kwargs = {'split': split, 'streaming': streaming}
    if subset:
        kwargs['name'] = subset
    
    ds = load_dataset(dataset_name, **kwargs)
    for sample in ds:
        yield sample


def convert_to_mds(
    sources: list,
    output_dir: str,
    text_key: str = 'text',
    compression: str = 'zstd',
    size_limit: int = 1 << 26,  # 64MB per shard
):
    """
    Convert multiple data sources to a single MDS dataset.
    
    Args:
        sources: List of source configs, each is a dict with:
            - type: 'jsonl', 'json', 'csv', 'tsv', 'txt', 'huggingface'
            - path: filepath (for local files) or dataset name (for HF)
            - text_key: column/key containing the text (default: 'text')
            - Additional kwargs depending on type
        output_dir: Where to write the MDS dataset
        text_key: Default text column name
        compression: Compression algorithm ('zstd', 'gzip', etc.)
        size_limit: Max shard size in bytes
    """
    
    # MDS columns - ModernBERT expects 'text' for raw text
    columns = {'text': 'str'}
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_samples = 0
    
    with MDSWriter(
        out=str(output_path),
        columns=columns,
        compression=compression,
        size_limit=size_limit
    ) as writer:
        
        for source in sources:
            source_type = source['type']
            source_text_key = source.get('text_key', text_key)
            
            print(f"\nProcessing: {source.get('path', source.get('dataset', 'unknown'))}")
            
            # Get iterator based on source type
            if source_type == 'jsonl':
                iterator = read_jsonl(source['path'])
            elif source_type == 'json':
                iterator = read_json(source['path'])
            elif source_type == 'csv':
                iterator = read_csv(source['path'], delimiter=',')
            elif source_type == 'tsv':
                iterator = read_csv(source['path'], delimiter='\t')
            elif source_type == 'txt':
                separator = source.get('doc_separator', '\n\n')
                iterator = read_txt(source['path'], doc_separator=separator)
            elif source_type == 'huggingface':
                iterator = read_huggingface(
                    dataset_name=source['dataset'],
                    split=source.get('split', 'train'),
                    text_column=source_text_key,
                    subset=source.get('subset'),
                    streaming=source.get('streaming', True)
                )
            else:
                raise ValueError(f"Unknown source type: {source_type}")
            
            # Write samples
            source_count = 0
            for sample in tqdm(iterator, desc=f"  {source_type}"):
                # Extract text from sample
                if isinstance(sample, str):
                    text = sample
                elif isinstance(sample, dict):
                    if source_text_key not in sample:
                        available_keys = list(sample.keys())
                        raise KeyError(
                            f"Text key '{source_text_key}' not found in sample from "
                            f"{source.get('path', source.get('dataset', 'unknown'))}. "
                            f"Available keys: {available_keys}. "
                            f"Set 'text_key' in your source config to the correct column name."
                        )
                    text = sample[source_text_key]
                else:
                    continue
                
                # Skip empty texts
                if not text or not text.strip():
                    continue
                
                writer.write({'text': text.strip()})
                source_count += 1
                total_samples += 1
            
            print(f"  Wrote {source_count:,} samples from this source")
    
    print(f"\n{'='*50}")
    print(f"Total samples written: {total_samples:,}")
    print(f"Output directory: {output_dir}")


# Example usage
if __name__ == '__main__':
    sources = [
        # SYNTHETIC corpus (S): output of the NeMo Curator pipeline on synthetic generations
        {
            'type': 'jsonl',
            # xxx = your scratch root; second xxx = output dir of nemo_pipeline_climate.py for S
            'path': '/home/xxx/data/xxx/nemo_curator/synthetic/final_pretrain_data.jsonl',
            'text_key': 'text'
        },
        # ACADEMIC corpus (A): output of the NeMo Curator pipeline on academic / journal text
        {
            'type': 'jsonl',
            # xxx = your scratch root; second xxx = output dir of nemo_pipeline_climate.py for A
            'path': '/home/xxx/data/xxx/nemo_curator/academic/final_pretrain_data.jsonl',
            'text_key': 'text'
        },
        # FineWeb-Edu climate-filtered subset (F): pushed to HF Hub by stream_filter_upload_fineweb.py
        {
            'type': 'huggingface',
            'dataset': 'xxx/fineweb-edu-climate',   # xxx = your HF org / username
            'split': 'train',
            'text_key': 'text',
            'streaming': True  # Stream to avoid downloading all at once
        },
        # Optional second HF source (e.g. additional climate corpus)
        {
            'type': 'huggingface',
            'dataset': 'xxx/another-climate-corpus', # xxx = your HF org / username; replace with whatever extra corpus you want to mix in
            # 'subset': '20220301.en',
            'split': 'train',
            'text_key': 'text',
            'streaming': True
        },
    ]

    convert_to_mds(
        sources=sources,
        # xxx = your scratch root; second xxx = output MDS dataset dir (matches `data_local` in the YAML configs)
        output_dir='/home/xxx/scratch/xxx',
        compression=None,
        size_limit=64 * 1024 * 1024  # 64MB shards
    )