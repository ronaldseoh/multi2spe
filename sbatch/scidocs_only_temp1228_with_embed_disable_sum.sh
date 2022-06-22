#!/bin/bash
#SBATCH -A pi_tbernard_umass_edu
#SBATCH --job-name=scidocs_only_cosine
#SBATCH -o sbatch_logs/stdout/scidocs_only_cosine_%j.txt
#SBATCH -e sbatch_logs/stderr/scidocs_only_cosine_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=gypsum-1080ti-ms
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"
conda activate qm

EXPERIMENT_ID_PREFIX=shard11_k-3_sum_embs_original-0-5+no_sum-0-5_extra_facet_alternate_layer_8_4-alternate_identity_common_random_cro
ss_entropy
EXPERIMENT_DATE="06-09"

python embed.py --pl-checkpoint-path /gypsum/work1/696ds-s21/bseoh/quartermaster/save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/checkpoints/last.ckpt \
                --data-path ../scidocs/data/paper_metadata_mag_mesh.json \
                --output /gypsum/work1/696ds-s21/bseoh/quartermaster/save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/cls_no_sum.jsonl --batch-size 4 \
                --debug_disable_sum_embs
