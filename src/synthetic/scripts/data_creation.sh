#!/bin/bash -l

#SBATCH --time=4:00:00
#SBATCH --nodes=1
##SBATCH --partition=nocapstor
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=a-infra01-1
#SBATCH --job-name=climate-data-creation
#SBATCH --output=climate-data-creation.log
#SBATCH --error=climate-data-creation.log


ANNOTATOR_MODEL="Qwen3-30B"
FULL_ANNOTATOR_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
ROOT="/iopsstor/scratch/cscs/jni/ClimateModernBERT"

VLLM_LOG="$ROOT/vllm_qwen3_30b.log"
CACHE_DIR="$ROOT/vllm_cache"
TIMEOUT=300
DEBUG=0
SEED=42
MAX_SEEDS=200
SAMPLES_PER_SEED=1
N_THREADS=128
HF_REPO="JingweiNi/climate-news-synthetic-seed-42"

export HF_HOME="/iopsstor/scratch/cscs/jni/hf_home"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TIKTOKEN_ENCODINGS_BASE="/iopsstor/scratch/cscs/jni/tiktoken_encodings"
huggingface-cli login --token $HUGGINGFACE_TOKEN

VLLM_CMD="vllm serve $FULL_ANNOTATOR_MODEL --tensor-parallel-size 4"

GENERATION_CMD="
INPUT_FILE_LIST=(
    \"$ROOT/seed/climatenews_2000_filtered.csv\"
    \"$ROOT/seed/climatenews_2005_filtered.csv\"
    \"$ROOT/seed/climatenews_2017_filtered.csv\"
    \"$ROOT/seed/climatenews_2022_filtered.csv\"
)

OUTPUT_FILE_LIST=(
    \"$ROOT/data/climatenews_2000_filtered_synthetic.jsonl\"
    \"$ROOT/data/climatenews_2005_filtered_synthetic.jsonl\"
    \"$ROOT/data/climatenews_2017_filtered_synthetic.jsonl\"
    \"$ROOT/data/climatenews_2022_filtered_synthetic.jsonl\"
)

for i in \${!INPUT_FILE_LIST[@]}; do
    python \"$ROOT/climatenew_vllm.py\" \
     --input_file \"\${INPUT_FILE_LIST[i]}\" \
     --output_file \"\${OUTPUT_FILE_LIST[i]}\" \
     --max_seeds $MAX_SEEDS \
     --samples_per_seed $SAMPLES_PER_SEED \
     --run_all_styles \
     --model_name \"$ANNOTATOR_MODEL\" \
     --seed $SEED \
     --cache_dir \"$CACHE_DIR\" \
     --n_threads $N_THREADS \
     --hf_repo \"$HF_REPO\"
done
"

ENV_CMD="pip install parse nltk sentence-transformers rouge_score  && pip uninstall -y numpy && pip install --no-cache-dir 'numpy==1.26.4'"

if [ $DEBUG -eq 0 ]; then
    LAUNCHING="srun --container-writable --environment=qwen3_next"
else
    LAUNCHING=""
fi

$LAUNCHING bash -lc "
  set -euo pipefail

  $ENV_CMD

  # 1) start vLLM in background
  $VLLM_CMD > \"$VLLM_LOG\" 2>&1 &
  VLLM_PID=\$!
  echo \"[INFO] vLLM PID=\$VLLM_PID\"

  # Ensure cleanup on any exit
  trap 'echo \"[CLEANUP] Stopping vLLM (\$VLLM_PID)\"; kill -TERM \$VLLM_PID 2>/dev/null || true; wait \$VLLM_PID 2>/dev/null || true' EXIT

  # 2) wait for readiness (timeout ${TIMEOUT}s)
  echo \"[WAIT] Watching $VLLM_LOG for readiness...\"
  if ! timeout $TIMEOUT bash -c '( tail -n0 -f \"$VLLM_LOG\" & ) | grep -q -- \"Application startup complete.\"'; then
    echo \"[ERROR] vLLM did not become ready within ${TIMEOUT}s\"
    exit 1
  fi
  echo \"[READY] vLLM is ready.\"

  # 3) run env setup + annotation
  echo \"[RUN] Launching annotation...\" 
  
  $GENERATION_CMD
  
  kill -TERM \$VLLM_PID 2>/dev/null || true
  wait \$VLLM_PID 2>/dev/null || true
  trap - EXIT
  exit 0
"
