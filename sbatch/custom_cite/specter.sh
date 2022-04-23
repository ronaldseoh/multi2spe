#!/bin/bash
#SBATCH --job-name=custom_cite_specter
#SBATCH -o sbatch_logs/stdout/custom_cite_specter_%j.txt
#SBATCH -e sbatch_logs/stderr/custom_cite_specter_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=2080ti-short
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"

EXPERIMENT_ID_PREFIX=specter
EXPERIMENT_DATE="02-02"

conda activate scidocs

python ../scidocs/scripts/run_custom_cite.py --user-citation ../quartermaster/save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation_custom_cite.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --data-path ~/my_scratch/scidocs-shard7-14 \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results_custom_cite.xlsx
