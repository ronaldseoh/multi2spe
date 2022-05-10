#!/bin/bash
#SBATCH --job-name=custom_cite_scincl_U_specter
#SBATCH -o sbatch_logs/stdout/custom_cite_scincl_U_specter_%j.txt
#SBATCH -e sbatch_logs/stderr/custom_cite_scincl_U_specter_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=cpu-long
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=2

eval "$(conda shell.bash hook)"

EXPERIMENT_ID_PREFIX=scincl_U_specter
EXPERIMENT_DATE="03-02"

conda activate scidocs

python ../scidocs/scripts/run_custom_cite.py --user-citation ../quartermaster/save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/user-citation_custom_cite.jsonl \
                      --val_or_test test \
                      --multifacet-behavior extra_linear \
                      --n-jobs 4 --cuda-device 0 \
                      --data-path /gypsum/scratch1/bseoh/scidocs-shard7 \
                      --results-save-path save_${EXPERIMENT_ID_PREFIX}_${EXPERIMENT_DATE}/results_custom_cite_cross_domain.xlsx
