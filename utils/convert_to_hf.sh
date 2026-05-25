#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=8096
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
hpc_notify "🚀 Started: MBert Conversion"
# python convert_to_hf.py --output-name MB_CPT_FWEDU_1Btok --output-dir /scratch/sraj/ --input-checkpoint /scratch/sraj/checkpoints/modernbert-base-context-extension-fineweb-1Btok/ep0-ba212-rank0.pt

# python convert_to_hf.py --output-name MB_CPT_ZYDA_ARXIV_2Btok --output-dir /scratch/sraj/ --input-checkpoint /scratch/sraj/checkpoints/modernbert-base-context-extension-zyda_arxiv/ep0-ba427-rank0.pt

# python convert_to_hf.py --output-name MB_CPT_ZYDA_ARXIV_2Btok_latestrank --output-dir /scratch/sraj/ --input-checkpoint /scratch/sraj/checkpoints/modernbert-base-context-extension-zyda_arxiv/latest-rank0.pt

# python convert_to_hf.py --output-name MB_CPT_ZYDA_ARXIV_FB_5Btok --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/checkpoints/modernbert-base-context-extension-zyda-arxiv-fineweb-5Btok/latest-rank0.pt

# python convert_to_hf.py --output-name MB_CPT_FWEB_10Btok --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/checkpoints/modernbert-base-context-extension-fineweb-10Btok/latest-rank0.pt

# python convert_to_hf.py --output-name MB_CPT_ARXIV_LR_DEC_FWEB_5Btok --output-dir /scratch/sraj/ --input-checkpoint ./checkpoints/arxiv-context-ext-fineweb-learning-rate-decay-5Btok/latest-rank0.pt

# python convert_to_hf.py --output-name MB_PT_FWEB_5Btok --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/checkpoints/modernbert-base-pretrain-fineweb-5Btok/latest-rank0.pt

# python convert_to_hf.py --output-name MB_CX2K_FWEB_10Btok --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/checkpoints/modernbert-base-context-ext2k-fineweb-10Btok/latest-rank0.pt

# python convert_to_hf.py --output-name MB_CX8K_CLIM_P_3EP --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/checkpoints/modernbert-base-context-ext8k-climate-para-3ep/latest-rank0.pt

# python convert_to_hf.py --output-name MB_CX8K_CLIM_P_5Btok --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/checkpoints/modernbert-base-context-ext8k-climate-para-5Btok/latest-rank0.pt

# python convert_to_hf.py --output-name MB_CX8K_CLIM_P_1EP --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/checkpoints/modernbert-base-context-ext8k-climate-para-1ep/latest-rank0.pt

# python convert_to_hf.py --output-name MB_CX8K_CLIM_P_NC_1EP --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/checkpoints/modernbert-base-context-ext8k-climate-para-NC-v1-1ep/latest-rank0.pt

# python convert_to_hf.py --output-name MB_CX8K_CLIM_P_NC_2EP --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/checkpoints/modernbert-base-context-ext8k-climate-para-NC-v1-2ep/latest-rank0.pt

# python convert_to_hf.py --output-name MB_CX8K_CLIM_P_NC_3EP --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/checkpoints/modernbert-base-context-ext8k-climate-para-NC-v1-3ep/latest-rank0.pt

# python convert_to_hf.py --output-name CMB_MICH_FWEdu_DEDUP_CX --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/MBcheckpoints/modernbert-base-context-ext8k-michael_fw_clim_dedup-3ep/latest-rank0.pt

# python convert_to_hf.py --output-name CMB_MICH_FWEdu_DEDUP_CX_LRD --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/MBcheckpointsLRD/michael_fw_clim_dedup_highprob_learning-rate-decay-3e/latest-rank0.pt

# python convert_to_hf.py --output-name CMB_MICH_FWEdu_CX --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/MBcheckpoints/modernbert-base-context-ext8k-michael_fw_clim-3ep/latest-rank0.pt

# python convert_to_hf.py --output-name CMB_MARK_CX --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/MBcheckpoints/modernbert-base-context-ext8k-markus-3ep/latest-rank0.pt

# python convert_to_hf.py --output-name CMB_MARK_WX_SYN_CX --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/MBcheckpoints/modernbert-base-context-ext8k-markus_wximpactbench_synthetic-3ep/latest-rank0.pt

# python convert_to_hf.py --output-name CMB_FWEdu_V2_CX --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/MBcheckpoints/modernbert-base-context-ext8k-sraj_finewebedu_v2-3ep/latest-rank0.pt

# python convert_to_hf.py --output-name CMB_WX_SYN_CX --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/MBcheckpoints/modernbert-base-context-ext8k-synthetic_wximpactbench-3ep/latest-rank0.pt


python convert_to_hf.py --output-name CMB_MARK_CX_LRD --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/MBcheckpointsLRD/modernbert-base-context-ext8k-markus-3ep_learning-rate-decay-3e/latest-rank0.pt

python convert_to_hf.py --output-name CMB_MARK_WX_SYN_CX_LRD --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/MBcheckpointsLRD/modernbert-base-context-ext8k-markus_wximpactbench_synthetic-3ep_learning-rate-decay-3e/latest-rank0.pt

python convert_to_hf.py --output-name CMB_FWEdu_V2_CX_LRD --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/MBcheckpointsLRD/modernbert-base-context-ext8k-sraj_finewebedu_v2-3ep_learning-rate-decay-3e/latest-rank0.pt

python convert_to_hf.py --output-name CMB_WX_SYN_CX_LRD --output-dir /scratch/sraj/ --input-checkpoint /home/sraj/scratch/MBcheckpointsLRD/modernbert-base-context-ext8k-synthetic_wximpactbench-3ep_learning-rate-decay-3e/latest-rank0.pt

hpc_notify "✅ Completed: MBert Conversion"


sbatch upload_to_hub.sh