#!/bin/bash
#SBATCH --job-name=scidocs_only_dot_normalize
#SBATCH -o sbatch_logs/stdout/scidocs_only_dot_normalize_%j.txt
#SBATCH -e sbatch_logs/stderr/scidocs_only_dot_normalize_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"
conda activate scidocs

EXPERIMENT_ID_PREFIX=k-3_common_nsp_cross_entropy
EXPERIMENT_DATE="09-24"

python ../scidocs/scripts/run.py --cls save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/cls.jsonl \
                      --user-citation save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation.jsonl \
                      --recomm save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/recomm.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --cls-svm \
                      --user-citation-metric "dot" \
                      --user-citation-normalize \
                      --data-path ../scidocs/data \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results_dot_normalize.xlsx
