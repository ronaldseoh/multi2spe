#!/bin/bash
#SBATCH --job-name=shard11-10-9_U_specter
#SBATCH -o sbatch_logs/stdout/shard11-10-9_U_specter_%j.txt
#SBATCH -e sbatch_logs/stderr/shard11-10-9_U_specter_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"
conda activate qm

EXPERIMENT_ID_PREFIX=shard11-10-9_U_specter
EXPERIMENT_DATE=`date +"%m-%d"`

python train.py --save_dir save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE} \
                --train_file /gypsum/scratch1/bseoh/20220328_shard_11_10_9/preprocessed/data-train.p --train_size 758740 \
                --val_file /gypsum/scratch1/bseoh/20220328_shard_11_10_9/preprocessed/data-val.p --val_size 234530 \
                --model_behavior 'specter' \
                --gpus 1 --num_workers 0 --fp16 \
                --batch_size 2 --grad_accum 16  --num_epochs 2 \
                --wandb

python embed.py --pl-checkpoint-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/checkpoints/last.ckpt \
                --data-path ../scidocs/data/paper_metadata_mag_mesh.json \
                --output save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/cls.jsonl --batch-size 4
                
python embed.py --pl-checkpoint-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/checkpoints/last.ckpt \
                --data-path ../scidocs/data/paper_metadata_recomm.json \
                --output save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/recomm.jsonl --batch-size 4
                
python embed.py --pl-checkpoint-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/checkpoints/last.ckpt \
                --data-path ../scidocs/data/paper_metadata_view_cite_read.json \
                --output save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation.jsonl --batch-size 4

conda deactivate
conda activate scidocs

python ../scidocs/scripts/run.py --cls ../quartermaster/save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/cls.jsonl \
                      --user-citation ../quartermaster/save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation.jsonl \
                      --recomm ../quartermaster/save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/recomm.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --cls-svm \
                      --data-path ../scidocs/data \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results.xlsx
