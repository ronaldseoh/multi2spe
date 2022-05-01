import transformers

import utils

if __name__ == '__main__':

    tokenizer = transformers.AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    
    dataset = utils.SciNclTripleDataset(
        triples_csv_path="~/my_scratch/scincl_dataset_wol/train_triples.csv", metadata_jsonl_path="/home/bseoh/my_scratch/scincl_dataset_wol/train_metadata.jsonl",
        tokenizer=tokenizer, use_cache=True,
        num_facets=3, use_cls_for_all_facets=False)
