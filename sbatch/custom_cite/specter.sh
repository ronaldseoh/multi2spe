#!/bin/bash
#SBATCH --job-name=custom_cite_specter
#SBATCH -o sbatch_logs/stdout/custom_cite_specter_%j.txt
#SBATCH -e sbatch_logs/stderr/custom_cite_specter_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=2080ti-short
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"
conda activate qm

EXPERIMENT_ID_PREFIX=specter
EXPERIMENT_DATE="02-02"

python embed.py --pl-checkpoint-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/checkpoints/last.ckpt \
                --data-path ~/my_scratch/scidocs-shard7_14_21/data_final.json \
                --output save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation_custom_cite_shard7_14_21.jsonl --batch-size 4

conda deactivate
conda activate scidocs

python ../scidocs/scripts/run_custom_cite.py --user-citation ../quartermaster/save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation_custom_cite_shard7_14_21.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --data-path ~/my_scratch/scidocs-shard7_14_21 \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results_custom_cite_shard7_14_21.xlsx
