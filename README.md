# Multi^2SPE

- `train.py` is our [PyTorch Ligthting](https://github.com/Lightning-AI/lightning)-based script for finetuning BERT models.
- `custom_bert.py` contains our modification of BERT architecture defined in [HuggingFace Transformers](https://github.com/huggingface/transformers/) library.
- `utils.py` contains PyTorch dataset classes for SPECTER and SciNCL dataset files.
- `s2orc` contains our codes for processing [S2ORC](https://github.com/allenai/s2orc) files into the format required by `specter`.
- `scincl` contains the version of the codes used for our experiments, originally provided by the SciNCL authors. Minor modifications has been added to allow importing their modules from our own codes.
- `specter` contains the version of the codes used for our experiments, originally provided by the SPECTER authors.
