#!/bin/bash
#SBATCH --job-name=custom_cite_scincl_U_k-3_sum_embs_original-0-5+no_sum-0-5_extra_facet_alternate_layer_8_4_identity_common_random_cross_entropy
#SBATCH -o sbatch_logs/stdout/custom_cite_scincl_U_k-3_sum_embs_original-0-5+no_sum-0-5_extra_facet_alternate_layer_8_4_identity_common_random_cross_entropy_%j.txt
#SBATCH -e sbatch_logs/stderr/custom_cite_scincl_U_k-3_sum_embs_original-0-5+no_sum-0-5_extra_facet_alternate_layer_8_4_identity_common_random_cross_entropy_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=cpu-long
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"

EXPERIMENT_ID_PREFIX=scincl_U_k-3_sum_embs_original-0-5+no_sum-0-5_extra_facet_alternate_layer_8_4_identity_common_random_cross_entropy
EXPERIMENT_DATE="03-16"

conda activate scidocs

python ../scidocs/scripts/run_custom_cite.py --user-citation ../quartermaster/save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation_custom_cite.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --user-citation-metric "cosine" \
                      --data-path /gypsum/scratch1/bseoh/scidocs-shard7-14 \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results_cosine_custom_cite_shard7-14.xlsx
