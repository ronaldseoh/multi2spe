#!/bin/bash
#SBATCH --job-name=U_k-5_debug_sum_embs_original-0-9+no_sum-0-1+mean-avg_word-0-05-input_emb_extra_facet_alternate_layer_8_4_identity_common_random_cross_entropy
#SBATCH -o sbatch_logs/stdout/U_k-5_debug_sum_embs_original-0-9+no_sum-0-1+mean-avg_word-0-05-input_emb_extra_facet_alternate_layer_8_4_identity_common_random_cross_entropy_%j.txt
#SBATCH -e sbatch_logs/stderr/U_k-5_debug_sum_embs_original-0-9+no_sum-0-1+mean-avg_word-0-05-input_emb_extra_facet_alternate_layer_8_4_identity_common_random_cross_entropy_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=gpu-long
#SBATCH --constraint=gypsum-titanx-ms
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"
conda activate qm

EXPERIMENT_ID_PREFIX=U_k-5_debug_sum_embs_original-0-9+no_sum-0-1+mean-avg_word-0-05-input_emb_extra_facet_alternate_layer_8_4_identity_common_random_cross_entropy
EXPERIMENT_DATE=`date +"%m-%d"`

python train.py --save_dir save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE} \
                --train_file /gypsum/scratch1/bseoh/original_data/train_shuffled.pkl --train_size 684100 \
                --val_file /gypsum/scratch1/bseoh/original_data/val_shuffled.pkl --val_size 145375 \
                --model_behavior 'quartermaster' --num_facets 5 \
                --add_extra_facet_layers_after 3 7 \
                --init_bert_layer_facet_layers 'identity' \
                --sum_into_single_embeddings 'training_and_inference' \
                --add_extra_facet_layers \
                --add_extra_facet_nonlinearity \
                --add_extra_facet_layers_alternate \
                --loss_config '[{"name": "original", "weight": 0.9, "loss_type": "bce", "margin": 1.0, "distance": "dot", "reduction": "mean", "reduction_multifacet": "max", "use_target_token_embs": false, "sum_into_single_embeddings": true}, {"name": "no_sum", "weight": 0.1, "loss_type": "bce", "margin": 1.0, "distance": "dot", "reduction": "mean", "reduction_multifacet": "max", "use_target_token_embs": false, "sum_into_single_embeddings": false}, {"name": "mean_and_word_emb", "weight": 0.05, "loss_type": "bce", "margin": 1.0, "distance": "dot", "reduction": "mean", "reduction_multifacet": "mean", "use_target_token_embs": true, "use_target_token_embs_input": true}]' \
                --gpus 1 --num_workers 0 --fp16 \
                --batch_size 2 --grad_accum 16  --num_epochs 2 \
                --seed 1945 \
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

python ../scidocs/scripts/run.py --cls save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/cls.jsonl \
                      --user-citation save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation.jsonl \
                      --recomm save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/recomm.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --cls-svm \
                      --data-path ../scidocs/data \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results.xlsx

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
