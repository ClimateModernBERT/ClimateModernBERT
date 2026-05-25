#!/bin/bash


source .venv/bin/activate
# uv pip install mergekit immutables
# ==============================================================================
#  FILL THESE IN
# ==============================================================================
export HF_TOKEN="...hf token here..."
HF_USERNAME="...hf username here..."
# ==============================================================================

set -euo pipefail

hf auth login --token "$HF_TOKEN" --add-to-git-credential

CONFIGS_DIR="./merge_configs"
OUTPUTS_DIR="./merge_outputs"
mkdir -p logs "$OUTPUTS_DIR"

# --------------------------------------------------------------------------
#  Define merges: config_file -> output_dir -> hf_repo_name
# --------------------------------------------------------------------------
declare -A MERGES=(
    # ["merge_all"]="Merge_CMB_MARK_CX_LRD_CMB_FWEdu_V2_FastText_CX_LRD_CMB_WX_SYN_CX_LRD"
    ["merge_drop_S"]="Merge_Drop_S"
    ["merge_drop_A"]="Merge_Drop_A"
    ["merge_drop_F"]="Merge_Drop_F"


    # --- TIES (density 0.5 and 0.7) ---
    ["merge_ties_d05"]="TIES_D05_A_F_S"
    ["merge_ties_d07"]="TIES_D07_A_F_S"

    # --- DARE-TIES (density 0.5 and 0.7) ---
    ["merge_dare_d05"]="DARE_TIES_D05_A_F_S"
    ["merge_dare_d07"]="DARE_TIES_D07_A_F_S"

    # --- Task Arithmetic (lambda 0.5 and 1.0) ---
    ["merge_ta_lambda05"]="TA_Lambda05_A_F_S"
    ["merge_ta_lambda10"]="TA_Lambda10_A_F_S"
)

# --------------------------------------------------------------------------
#  Run merges and push
# --------------------------------------------------------------------------
for config_name in "${!MERGES[@]}"; do
    repo_name="${MERGES[$config_name]}"
    config_path="${CONFIGS_DIR}/${config_name}.yaml"
    output_path="${OUTPUTS_DIR}/${config_name}"

    echo "=============================================="
    echo " Merging: ${config_name}"
    echo " Config:  ${config_path}"
    echo " Output:  ${output_path}"
    echo " Repo:    ${HF_USERNAME}/${repo_name}"
    echo "=============================================="

    # Skip if output already exists (allows resuming after a failure)
    if [ -d "${output_path}" ] && [ -f "${output_path}/config.json" ]; then
        echo "  -> Output already exists, skipping merge step."
    else
        rm -rf "${output_path}"
        mergekit-yaml "${config_path}" "${output_path}" \
            --copy-tokenizer \
            --lazy-unpickle \
            --out-shard-size 2B
    fi

    echo "  -> Pushing to ${HF_USERNAME}/${repo_name} ..."
    hf upload "${HF_USERNAME}/${repo_name}" "${output_path}" . \
        --commit-message "Linear merge: ${config_name}"

    echo "  -> Done: ${config_name}"
    echo ""
done

echo "=============================================="
echo " All merges complete and pushed."
echo "=============================================="
