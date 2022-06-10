#!/bin/bash
#SBATCH --job-name=custom_cite_scincl-wol_k-3_debug_sum_embs_original-0-9+no_sum-0-1_extra_facet_alternate_layer_8_4-alternate_identity_common_random_cross_entropy
#SBATCH -o sbatch_logs/stdout/custom_cite_scincl-wol_k-3_debug_sum_embs_original-0-9+no_sum-0-1_extra_facet_alternate_layer_8_4-alternate_identity_common_random_cross_entropy_%j.txt
#SBATCH -e sbatch_logs/stderr/custom_cite_scincl-wol_k-3_debug_sum_embs_original-0-9+no_sum-0-1_extra_facet_alternate_layer_8_4-alternate_identity_common_random_cross_entropy_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=2080ti-short
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"

EXPERIMENT_ID_PREFIX=scincl-wol_k-3_debug_sum_embs_original-0-9+no_sum-0-1_extra_facet_alternate_layer_8_4-alternate_identity_common_random_cross_entropy
EXPERIMENT_DATE="05-04"

conda activate scidocs

python ../scidocs/scripts/run_custom_cite.py --user-citation ../quartermaster/save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation_custom_cite.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --user-citation-metric "cosine" \
                      --data-path ~/my_scratch/scidocs-shard7 \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results_cosine_custom_cite_cross_domain.xlsx
