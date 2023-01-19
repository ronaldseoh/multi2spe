# Multi^2SPE

- `train.py` is our [PyTorch Lightning](https://github.com/Lightning-AI/lightning)-based script for finetuning BERT models.
- `embed.py` is the script for producing paper embeddings to be fed into the SciDocs testing suite.
- `custom_bert.py` contains our modification of BERT architecture defined in [HuggingFace Transformers](https://github.com/huggingface/transformers/) library.
- `utils.py` contains PyTorch dataset classes for SPECTER and SciNCL dataset files.

In order to train your own model, please use the following command:

```bash
python train.py --save_dir save \
                --train_file train.pkl --train_size 684100 \
                --val_file val.pkl --val_size 145375 \
                --model_behavior 'quartermaster' --num_facets 3 \
                --add_extra_facet_layers_after 3 7 \
                --init_bert_layer_facet_layers 'identity' \
                --sum_into_single_embeddings 'training_and_inference' \
                --add_extra_facet_layers \
                --add_extra_facet_nonlinearity \
                --add_extra_facet_layers_alternate \
                --add_bert_layer_facet_layers_alternate \
                --loss_config '[{"name": "original", "weight": 0.9, "loss_type": "bce", "margin": 1.0, "distance": "dot", "reduction": "mean", "reduction_multifacet": "max", "use_target_token_embs": false, "sum_into_single_embeddings": true}, {"name": "no_sum", "weight": 0.1, "loss_type": "bce", "margin": 1.0, "distance": "dot", "reduction": "mean", "reduction_multifacet": "max", "use_target_token_embs": false, "sum_into_single_embeddings": false}]' \
                --gpus 1 --num_workers 0 --fp16 \
                --batch_size 2 --grad_accum 16  --num_epochs 2 \
                --seed 1991 \
                --wandb
```

For producing paper embeddings for SciDocs, please run the following:

```bash
python embed.py --pl-checkpoint-path save/checkpoints/last.ckpt \
                --data-path scidocs/data/paper_metadata_mag_mesh.json \
                --output save/cls.jsonl --batch-size 4
                
python embed.py --pl-checkpoint-path save/checkpoints/last.ckpt \
                --data-path scidocs/data/paper_metadata_recomm.json \
                --output save/recomm.jsonl --batch-size 4
                
python embed.py --pl-checkpoint-path save/checkpoints/last.ckpt \
                --data-path scidocs/data/paper_metadata_view_cite_read.json \
                --output save/user-citation.jsonl --batch-size 4
```

Please refer to our [`s2orc`](https://anonymous.4open.science/r/s2orc-7F1E/) repo for the instructions on producing multi-domain SPECTER training datasets and Multi-SciDocs datasets. After getting your own multi-domain SPECTER datasets, you could use them for training by setting the paths appropriately in `train_file` and `val_file` options for the training command above.

Please refer to our [`scidocs`](https://anonymous.4open.science/r/scidocs-A231/) repo for the instructions on running (Multi-)SciDocs benchmark.
