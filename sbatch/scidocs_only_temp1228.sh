#!/bin/bash
#SBATCH --job-name=scidocs_only_cosine
#SBATCH -o sbatch_logs/stdout/scidocs_only_cosine_%j.txt
#SBATCH -e sbatch_logs/stderr/scidocs_only_cosine_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=gpu-long
#SBATCH --constraint=ials_gigabyte_gpu_2020
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"


EXPERIMENT_ID_PREFIX=shard11_U_k-3_sum_embs_original-0-9+no_sum-0-1_extra_facet_alternate_layer_8_4-alternate_identity_common_random_cross_entropy
EXPERIMENT_DATE="07-14"

conda activate scidocs

python ../scidocs/scripts/run_custom_cite.py --user-citation save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation_custom_cite.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --user-citation-metric "cosine" \
                      --data-path /gypsum/scratch1/bseoh/scidocs-shard7-full \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results_cosine_custom_cite-full.xlsx
