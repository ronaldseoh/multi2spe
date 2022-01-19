#!/bin/bash
#SBATCH --job-name=scidocs_only_cosine
#SBATCH -o sbatch_logs/stdout/scidocs_only_cosine_%j.txt
#SBATCH -e sbatch_logs/stderr/scidocs_only_cosine_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=gpu-long
#SBATCH --constraint=ials_gigabyte_gpu_2020
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"


EXPERIMENT_ID_PREFIX=U_k-3_sum_embs_original-0-9+no_sum-0-1+max-kmeans_word-0-05_extra_facet_alternate_layer_8_4_identity_common_random_cross_entropy
EXPERIMENT_DATE="01-17"


conda activate scidocs

python ../scidocs/scripts/run.py --cls save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/cls.jsonl \
                      --user-citation save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation.jsonl \
                      --recomm save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/recomm.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --cls-svm \
                      --data-path ../scidocs/data \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results.xlsx

python ../scidocs/scripts/run.py --cls save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/cls.jsonl \
                      --user-citation save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation.jsonl \
                      --recomm save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/recomm.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --cls-svm \
                      --user-citation-metric "cosine" \
                      --data-path ../scidocs/data \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results_cosine.xlsx
