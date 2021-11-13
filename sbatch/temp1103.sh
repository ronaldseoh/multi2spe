#!/bin/bash
#SBATCH --job-name=scidocs_only_cosine
#SBATCH -o sbatch_logs/stdout/scidocs_only_cosine_%j.txt
#SBATCH -e sbatch_logs/stderr/scidocs_only_cosine_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"
conda activate qm

EXPERIMENT_ID_PREFIX=multilosstest2
EXPERIMENT_DATE="10-14"

python embed.py --pl-checkpoint-path ${EXPERIMENT_ID_PREFIX}/checkpoints/last.ckpt \
                --data-path ../scidocs/data/paper_metadata_mag_mesh.json \
                --output ${EXPERIMENT_ID_PREFIX}/cls.jsonl --batch-size 4
                
python embed.py --pl-checkpoint-path ${EXPERIMENT_ID_PREFIX}/checkpoints/last.ckpt \
                --data-path ../scidocs/data/paper_metadata_recomm.json \
                --output ${EXPERIMENT_ID_PREFIX}/recomm.jsonl --batch-size 4
                
python embed.py --pl-checkpoint-path ${EXPERIMENT_ID_PREFIX}/checkpoints/last.ckpt \
                --data-path ../scidocs/data/paper_metadata_view_cite_read.json \
                --output ${EXPERIMENT_ID_PREFIX}/user-citation.jsonl --batch-size 4

conda deactivate
conda activate scidocs

python ../scidocs/scripts/run.py --cls ${EXPERIMENT_ID_PREFIX}/cls.jsonl \
                      --user-citation ${EXPERIMENT_ID_PREFIX}/user-citation.jsonl \
                      --recomm ${EXPERIMENT_ID_PREFIX}/recomm.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --cls-svm \
                      --data-path ../scidocs/data \
                      --results-save-path ${EXPERIMENT_ID_PREFIX}/results.xlsx

python ../scidocs/scripts/run.py --cls ${EXPERIMENT_ID_PREFIX}/cls.jsonl \
                      --user-citation ${EXPERIMENT_ID_PREFIX}/user-citation.jsonl \
                      --recomm ${EXPERIMENT_ID_PREFIX}/recomm.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --cls-svm \
                      --user-citation-metric "cosine" \
                      --data-path ../scidocs/data \
                      --results-save-path ${EXPERIMENT_ID_PREFIX}/results_cosine.xlsx