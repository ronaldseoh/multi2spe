#!/bin/bash
#SBATCH --job-name=scincl_k-3_sum_embs-mean_original-0-5+no_sum-0-5+mean-avg_word-0-05_extra_facet_alternate_layer_8_4_identity_common_random_margin
#SBATCH -o sbatch_logs/stdout/scincl_k-3_sum_embs-mean_original-0-5+no_sum-0-5+mean-avg_word-0-05_extra_facet_alternate_layer_8_4_identity_common_random_margin_%j.txt
#SBATCH -e sbatch_logs/stderr/scincl_k-3_sum_embs-mean_original-0-5+no_sum-0-5+mean-avg_word-0-05_extra_facet_alternate_layer_8_4_identity_common_random_margin_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"
conda activate qm

EXPERIMENT_ID_PREFIX=scincl_k-3_sum_embs-mean_original-0-5+no_sum-0-5+mean-avg_word-0-05_extra_facet_alternate_layer_8_4_identity_common_random_margin
EXPERIMENT_DATE=`date +"%m-%d"`

python train.py --save_dir save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE} \
                --train_file ~/my_scratch/scincl_dataset/train_triples.csv --train_metadata_file ~/my_scratch/scincl_dataset/train_metadata.jsonl --train_file_from_scincl --train_size 684100 \
                --val_file ~/my_scratch/original_data/val_shuffled.pkl --val_size 145375 \
                --model_behavior 'quartermaster' --num_facets 3 \
                --add_extra_facet_layers_after 3 7 \
                --init_bert_layer_facet_layers 'identity' \
                --sum_into_single_embeddings 'training_and_inference' \
                --sum_into_single_embeddings_behavior 'mean' \
                --add_extra_facet_layers \
                --add_extra_facet_nonlinearity \
                --add_extra_facet_layers_alternate \
                --loss_config '[{"name": "original", "weight": 0.5, "loss_type": "margin", "margin": 1.0, "distance": "l2-norm", "reduction": "mean", "reduction_multifacet": "min", "use_target_token_embs": false, "sum_into_single_embeddings": true}, {"name": "no_sum", "weight": 0.5, "loss_type": "margin", "margin": 1.0, "distance": "l2-norm", "reduction": "mean", "reduction_multifacet": "min", "use_target_token_embs": false, "sum_into_single_embeddings": false}, {"name": "mean_and_word_emb", "weight": 0.05, "loss_type": "margin", "margin": 1.0, "distance": "l2-norm", "reduction": "mean", "reduction_multifacet": "mean", "use_target_token_embs": true}]' \
                --gpus 1 --num_workers 0 --fp16 \
                --batch_size 2 --grad_accum 16  --num_epochs 2 \
                --seed 1991 \
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
