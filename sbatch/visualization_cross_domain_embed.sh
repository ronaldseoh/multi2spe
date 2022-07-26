#!/bin/bash
#SBATCH -A pi_tbernard_umass_edu
#SBATCH --job-name=scidocs_only_cosine
#SBATCH -o sbatch_logs/stdout/scidocs_only_cosine_%j.txt
#SBATCH -e sbatch_logs/stderr/scidocs_only_cosine_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=gypsum-2080ti
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"
conda activate qm

EXPERIMENT_ID_PREFIX=shard11_U_specter
EXPERIMENT_DATE="03-29"

python embed.py --pl-checkpoint-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/checkpoints/last.ckpt \
                --data-path ../20220721_shard_3_cross/metadata.json \
                --output ../20220721_shard_3_cross/specter_embeddings_single.jsonl --batch-size 4
