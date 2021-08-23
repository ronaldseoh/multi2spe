#!/bin/bash
#SBATCH --job-name=k-1
#SBATCH -o sbatch_logs/stdout/k-1_%j.txt
#SBATCH -e sbatch_logs/stderr/k-1_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"
conda activate quartermaster

EXPERIMENT_ID_PREFIX=k-1
EXPERIMENT_DATE=`date +"%m-%d"`

python train.py --save_dir save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE} \
                --train_file 0819_shard_0_1/data-train.p --train_size 670470 \
                --val_file 0819_shard_0_1/data-val.p --val_size 168015 \
                --model_behavior 'quartermaster' --num_facets 1 --add_extra_facet_layers --loss_reduction_multifacet 'min' \
                --gpus 1 --num_workers 0 --fp16 \
                --batch_size 2 --grad_accum 16  --num_epochs 2 \
                --wandb
