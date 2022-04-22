#!/bin/bash
#SBATCH --job-name=custom_cite
#SBATCH -o sbatch_logs/stdout/custom_cite_%j.txt
#SBATCH -e sbatch_logs/stderr/custom_cite_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --constraint=ials_gigabyte_gpu_2020
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"
conda activate qm

EXPERIMENT_ID_PREFIX=specter
EXPERIMENT_DATE="09-09"
                
python embed.py --pl-checkpoint-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/checkpoints/last.ckpt \
                --data-path /gypsum/scratch1/bseoh/scidocs-shard7/data_final.json \
                --output save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation_custom_cite.jsonl --batch-size 4

conda deactivate
conda activate scidocs

python ../scidocs/scripts/run_custom_cite.py --user-citation ../quartermaster/save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation_custom_cite.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --data-path /gypsum/scratch1/bseoh/scidocs-shard7 \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results_custom_cite.xlsx
