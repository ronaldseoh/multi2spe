#!/bin/bash
#SBATCH --job-name=custom_cite_scincl_k-3_sum_embs_original-0-5+no_sum-0-5_extra_facet_alternate_layer_8_4_identity_common_random_cross_entropy
#SBATCH -o sbatch_logs/stdout/custom_cite_scincl_k-3_sum_embs_original-0-5+no_sum-0-5_extra_facet_alternate_layer_8_4_identity_common_random_cross_entropy_%j.txt
#SBATCH -e sbatch_logs/stderr/custom_cite_scincl_k-3_sum_embs_original-0-5+no_sum-0-5_extra_facet_alternate_layer_8_4_identity_common_random_cross_entropy_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=2080ti-short
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"
conda activate qm

EXPERIMENT_ID_PREFIX=scincl_k-3_sum_embs_original-0-5+no_sum-0-5_extra_facet_alternate_layer_8_4_identity_common_random_cross_entropy
EXPERIMENT_DATE="03-17"

python embed.py --pl-checkpoint-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/checkpoints/last.ckpt \
                --data-path ~/my_scratch/scidocs-shard7_14_21/data_final.json \
                --output save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation_custom_cite_shard7_14_21.jsonl --batch-size 4

conda deactivate
conda activate scidocs

python ../scidocs/scripts/run_custom_cite.py --user-citation ../quartermaster/save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation_custom_cite_shard7_14_21.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --user-citation-metric "cosine" \
                      --data-path ~/my_scratch/scidocs-shard7_14_21 \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results_cosine_custom_cite_shard7_14_21.xlsx
