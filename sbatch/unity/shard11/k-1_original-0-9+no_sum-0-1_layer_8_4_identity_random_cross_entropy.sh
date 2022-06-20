#!/bin/bash
#SBATCH --job-name=shard11_U_k-1_original-0-9+no_sum-0-1_layer_8_4_identity_random_cross_entropy
#SBATCH -o sbatch_logs/stdout/shard11_U_k-1_original-0-9+no_sum-0-1_layer_8_4_identity_random_cross_entropy_%j.txt
#SBATCH -e sbatch_logs/stderr/shard11_U_k-1_original-0-9+no_sum-0-1_layer_8_4_identity_random_cross_entropy_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=gpu-long
#SBATCH --constraint=ials_gigabyte_gpu_2020
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"
conda activate qm

EXPERIMENT_ID_PREFIX=shard11_U_k-1_original-0-9+no_sum-0-1_layer_8_4_identity_random_cross_entropy
EXPERIMENT_DATE=`date +"%m-%d"`

python train.py --save_dir save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE} \
                --train_file /gypsum/scratch1/bseoh/20220327_shard_11/preprocessed/data-train.p --train_size 252640 \
                --val_file /gypsum/scratch1/bseoh/20220327_shard_11/preprocessed/data-val.p --val_size 77900 \
                --model_behavior 'quartermaster' --num_facets 1 \
                --add_extra_facet_layers_after 3 7 \
                --init_bert_layer_facet_layers 'identity' \
                --add_extra_facet_layers \
                --add_extra_facet_nonlinearity \
                --loss_config '[{"name": "original", "weight": 0.9, "loss_type": "bce", "margin": 1.0, "distance": "dot", "reduction": "mean", "reduction_multifacet": "max", "use_target_token_embs": false, "sum_into_single_embeddings": true}, {"name": "no_sum", "weight": 0.1, "loss_type": "bce", "margin": 1.0, "distance": "dot", "reduction": "mean", "reduction_multifacet": "max", "use_target_token_embs": false, "sum_into_single_embeddings": false}]' \
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

python embed.py --pl-checkpoint-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/checkpoints/last.ckpt \
                --data-path /gypsum/scratch1/bseoh/scidocs-shard7/data_final.json \
                --output save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation_custom_cite.jsonl --batch-size 4

conda deactivate
conda activate scidocs

python ../scidocs/scripts/run.py --cls save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/cls.jsonl \
                      --user-citation save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation.jsonl \
                      --recomm save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/recomm.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --cls-svm \
                      --user-citation-metric "cosine" \
                      --data-path ../scidocs/data \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results_cosine.xlsx

python ../scidocs/scripts/run_custom_cite.py --user-citation ../quartermaster/save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation_custom_cite.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --user-citation-metric "cosine" \
                      --data-path /gypsum/scratch1/bseoh/scidocs-shard7 \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results_cosine_custom_cite.xlsx

python ../scidocs/scripts/run_custom_cite.py --user-citation save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation_custom_cite.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --data-path /gypsum/scratch1/bseoh/scidocs-shard7-full \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results_cosine_custom_cite-full.xlsx