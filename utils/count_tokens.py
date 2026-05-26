import json, glob

total_bytes = 0
total_samples = 0
# xxx = your scratch root; the inner xxx is the MDS dataset directory you converted with convert_any_to_mds.py
for idx_path in glob.glob('/home/xxx/scratch/xxx/train/index.json', recursive=True):
    with open(idx_path) as f:
        meta = json.load(f)
    for shard in meta['shards']:
        total_bytes += shard['raw_data']['bytes']
        total_samples += shard['samples']

print(f"{total_samples:,} documents, {total_bytes / 1e9:.2f} GB")
print(f"~{int(total_bytes / 4):,} tokens (rough English BPE estimate)")