#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=16384
#SBATCH --time=01:00:00
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.err

module load dev2025a 

export HF_HOME="/scratch/xxx/"
export TRITON_CACHE_DIR="/scratch/xxx/tritoncache"
export TRITON_HOME="/scratch/xxx/tritoncache"
export VLLM_CACHE_ROOT="/scratch/xxx/vllmcache"

source .venv/bin/activate

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

BASE_PATH="/home/xxx/scratch"      # xxx = your scratch root containing the HF-format checkpoint dirs from convert_to_hf.sh
HF_NAMESPACE="xxx"                 # xxx = your HF username / org that will own the uploaded model repos
MAKE_PRIVATE="false"
DELETE_EXISTING="false"

MODELS=(
	"CMB_A:CMB_A"
	"CMB_F:CMB_F"
	"CMB_S:CMB_S"
)


COLLECTION_SLUGS=( "xxx" )         # xxx = HF collection slug to add the uploaded repos to (leave empty to skip)

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

# `hpc_notify` is a cluster-local helper (Slack notifier). Drop or replace with your own:
# hpc_notify "✅ Completed: Model Upload"