# Multi^2SPE

- `train.py` is our [PyTorch Ligthting](https://github.com/Lightning-AI/lightning)-based script for finetuning BERT models.
- `custom_bert.py` contains our modification of BERT architecture defined in [HuggingFace Transformers](https://github.com/huggingface/transformers/) library.
- `utils.py` contains PyTorch dataset classes for SPECTER and SciNCL dataset files.

Please refer to our [`s2orc`](https://anonymous.4open.science/r/s2orc-7F1E/) repo for the instructions on producing multi-domain SPECTER training data and Multi-SciDocs data.
