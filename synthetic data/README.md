## Synthetic data generation scripts

This folder contains scripts to generate synthetic climate-related text from different seed datasets.

- `climatenew.py`: generate synthetic climate news from CSV seeds in `../seed`.

### Basic usage
Run from the project root, for example:

```bash
python synthetic\ data/climatenew.py \
  --input_file ../seed/climatenews_2000_filtered.csv \
  --output_file climatenews_syn.jsonl \
  --max_seeds 100 \
  --samples_per_seed 1 \
  --run_all_styles
```

All other scripts share the same arguments but take `--input_dir` instead of `--input_file` and default to their corresponding seed folders.
