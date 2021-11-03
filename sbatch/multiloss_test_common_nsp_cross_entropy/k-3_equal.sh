#!/bin/bash
#SBATCH --job-name=k-3_equal_common_nsp_cross_entropy
#SBATCH -o sbatch_logs/stdout/k-3_equal_common_nsp_cross_entropy_%j.txt
#SBATCH -e sbatch_logs/stderr/k-3_equal_common_nsp_cross_entropy_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"
conda activate qm

EXPERIMENT_ID_PREFIX=k-3_equal_common_nsp_cross_entropy
EXPERIMENT_DATE=`date +"%m-%d"`

python train.py --save_dir save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE} \
                --train_file ~/original_data/train_shuffled.pkl --train_size 684100 \
                --val_file ~/original_data/val_shuffled.pkl --val_size 145375 \
                --model_behavior 'quartermaster' --num_facets 3 \
                --add_extra_facet_layers --add_extra_facet_layers_initialize_with_nsp_weights \
                --add_extra_facet_nonlinearity \
                --loss_config '[{"name": "original", "weight": 1, "loss_type": "bce", "margin": 1.0, "distance": "dot", "reduction": "mean", "reduction_multifacet": "max", "use_target_token_embs": false}, {"name": "mean_and_word_emb", "weight": 1, "loss_type": "bce", "margin": 1.0, "distance": "dot", "reduction": "mean", "reduction_multifacet": "mean", "use_target_token_embs": true}, {"name": "mean_and_mean", "weight": 1, "loss_type": "bce", "margin": 1.0, "distance": "dot", "reduction": "mean", "reduction_multifacet": "mean", "use_target_token_embs": false}]' \
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
