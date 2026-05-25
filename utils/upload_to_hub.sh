#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=16384
#SBATCH --time=01:00:00
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.err

module load dev2025a 
source ~/hpc_notify.sh

export HF_HOME="/scratch/sraj/"
export TRITON_CACHE_DIR="/scratch/sraj/tritoncache"
export TRITON_HOME="/scratch/sraj/tritoncache"
export VLLM_CACHE_ROOT="/scratch/sraj/vllmcache"

source .venv/bin/activate
hpc_notify "🚀 Started: Model Upload"

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="./upload_to_hub.py"
TOKEN_FILE="./hf_token.txt"

if [[ ! -f "${TOKEN_FILE}" ]]; then
	echo "Token file not found at ${TOKEN_FILE}." >&2
	exit 1
fi

read -r HF_TOKEN < "${TOKEN_FILE}"
HF_TOKEN="${HF_TOKEN//$'\r'}"
HF_TOKEN="${HF_TOKEN//$'\n'}"
if [[ -z "${HF_TOKEN}" ]]; then
	echo "Token file ${TOKEN_FILE} is empty." >&2
	exit 1
fi
export HF_TOKEN

BASE_PATH="/home/sraj/scratch"
HF_NAMESPACE="sraj"
MAKE_PRIVATE="false"
DELETE_EXISTING="false"

# MODELS=(
# 	"MB_CPT_ZYDA_ARXIV_FB_5Btok:MB_CPT_ZYDA_ARXIV_ALL_FB_5Btok"
#     "MB_CPT_ZYDA_ARXIV_2Btok:MB_CPT_ZYDA_ARXIV_2Btok"
#     "MB_CPT_FWEDU_1Btok:MB_CPT_FWEDU_1Btok"
# 	"MB_CPT_FWEB_10Btok:MB_CPT_FWEDU_10Btok"
# )
# MODELS=(
# 	"MB_CPT_FWEB_10Btok:MB_CPT_FWEDU_10Btok"
# 	"MB_CPT_ARXIV_LR_DEC_FWEB_5Btok:MB_CPT_ARXIV_LR_DEC_FWEB_5Btok"
# )
# MODELS=(
# 	"MB_PT_FWEB_5Btok:MB_PT_FWEB_5Btok"
# 	"MB_CX2K_FWEB_10Btok:MB_CX2K_FWEB_10Btok"
# )
# MODELS=(
# 	"MB_CX8K_CLIM_P_3EP:MB_CX8K_CLIM_P_3EP"
# )
# MODELS=(
# 	"MB_CX8K_CLIM_P_5Btok:MB_CX8K_CLIM_P_5Btok"
# 	"MB_CX8K_CLIM_P_1EP:MB_CX8K_CLIM_P_1EP"
# )
# MODELS=(
# 	"MB_CX8K_CLIM_P_NC_1EP:MB_CX8K_CLIM_P_NCv1_1EP"
# 	"MB_CX8K_CLIM_P_NC_2EP:MB_CX8K_CLIM_P_NCv1_2EP"
# 	"MB_CX8K_CLIM_P_NC_3EP:MB_CX8K_CLIM_P_NCv1_3EP"
# )

# MODELS=(
# 	"CMB_MICH_FWEdu_DEDUP_CX:CMB_MICH_FWEdu_DEDUP_CX"
# 	"CMB_MICH_FWEdu_DEDUP_CX_LRD:CMB_MICH_FWEdu_DEDUP_CX_LRD"
# )

# MODELS=(
# 	"CMB_MICH_FWEdu_CX:CMB_MICH_FWEdu_CX"
# )
# "CMB_MARK_WX_SYN_CX:CMB_MARK_WX_SYN_CX"
# MODELS=(
# 	"CMB_MARK_CX:CMB_MARK_CX"
# 	"CMB_FWEdu_V2_CX:CMB_FWEdu_V2_CX"
# 	"CMB_WX_SYN_CX:CMB_WX_SYN_CX"
# )
MODELS=(
	"CMB_MARK_CX_LRD:CMB_MARK_CX_LRD"
	"CMB_MARK_WX_SYN_CX_LRD:CMB_MARK_WX_SYN_CX_LRD"
	"CMB_FWEdu_V2_CX_LRD:CMB_FWEdu_V2_CX_LRD"
	"CMB_WX_SYN_CX_LRD:CMB_WX_SYN_CX_LRD"
	"CMB_MARK_WX_SYN_CX:CMB_MARK_WX_SYN_CX"
	"CMB_FWEdu_V2_CX:CMB_FWEdu_V2_CX"
	"CMB_WX_SYN_CX:CMB_WX_SYN_CX"
)
MODELS=(
	"CMB_MARK_CX:CMB_MARK_CX"
)

COLLECTION_SLUGS=( "CMB" )
# COLLECTION_SLUGS=( "OMB" )

if [[ ${#MODELS[@]} -eq 0 ]]; then
	echo "No model directories specified in MODELS array." >&2
	exit 1
fi

if [[ ! -x "${PYTHON_SCRIPT}" ]]; then
	chmod +x "${PYTHON_SCRIPT}"
fi

for entry in "${MODELS[@]}"; do
	if [[ -z "${entry}" ]]; then
		continue
	fi

	IFS=":" read -r relative_path explicit_repo <<< "${entry}"

	if [[ -z "${relative_path}" ]]; then
		echo "Skipping blank model entry." >&2
		continue
	fi

	model_path="${BASE_PATH%/}/${relative_path}"
	if [[ ! -d "${model_path}" ]]; then
		echo "Directory not found: ${model_path}" >&2
		continue
	fi

	repo_suffix="${explicit_repo:-$(basename "${relative_path}")}"
	repo_id="${HF_NAMESPACE}/${repo_suffix}"

	echo "Uploading ${model_path} -> ${repo_id}" >&2

	python_args=(
		"--folder-path" "${model_path}"
		"--repo-id" "${repo_id}"
		"--commit-message" "Upload ${repo_suffix}"
	)

	if [[ "${MAKE_PRIVATE,,}" == "true" ]]; then
		python_args+=("--private")
	fi

	if [[ "${DELETE_EXISTING,,}" == "true" ]]; then
		python_args+=("--delete-existing")
	fi

	if [[ ${#COLLECTION_SLUGS[@]} -gt 0 ]]; then
		python_args+=("--collection-owner" "${HF_NAMESPACE}")
		for slug in "${COLLECTION_SLUGS[@]}"; do
			python_args+=("--collection" "${slug}")
		done
	fi

	python "${PYTHON_SCRIPT}" "${python_args[@]}"
done

hpc_notify "✅ Completed: Model Upload"