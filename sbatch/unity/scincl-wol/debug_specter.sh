#!/bin/bash
#SBATCH --job-name=scincl-wol_U_debug_specter
#SBATCH -o sbatch_logs/stdout/scincl-wol_U_debug_specter_%j.txt
#SBATCH -e sbatch_logs/stderr/scincl-wol_U_debug_specter_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=gypsum-titanx-ms
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"
conda activate qm

EXPERIMENT_ID_PREFIX=scincl-wol_U_debug_specter
EXPERIMENT_DATE=`date +"%m-%d"`

python train.py --save_dir save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE} \
                --train_file /gypsum/scratch1/bseoh/scincl_dataset_wol/train_triples.csv --train_metadata_file /gypsum/scratch1/bseoh/scincl_dataset_wol/train_metadata.jsonl --train_file_from_scincl --train_size 684100 \
                --val_file /gypsum/scratch1/bseoh/original_data/val_shuffled.pkl --val_size 145375 \
                --model_behavior 'specter' \
                --gpus 1 --num_workers 0 --fp16 \
                --batch_size 2 --grad_accum 16  --num_epochs 2 \
                --seed 1783 \
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
