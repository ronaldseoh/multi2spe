import argparse
import collections
import json

import torch
import transformers

import utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file')
    parser.add_argument('--train_size', type=int)

    parser.add_argument('--pretrained_model_name', default="allenai/scibert_scivocab_uncased", type=str)

    parser.add_argument('--output')

    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.pretrained_model_name)

    dataset = utils.IterableDataSetMultiWorker(file_path=args.train_file, tokenizer=tokenizer, size=args.train_size)

    main_counter = collections.Counter()

    for d in dataset:
        # Source paper
        source_input_ids = d[0]["input_ids"]
        source_special_tokens_mask = d[0]['special_tokens_mask']

        source_idx_non_special_tokens = torch.nonzero(~(source_special_tokens_mask.type(torch.bool))).flatten()
        source_input_ids_non_special_tokens = source_input_ids[source_idx_non_special_tokens]
        source_input_ids_non_special_tokens = source_input_ids_non_special_tokens.tolist()

        source_input_ids_counter = collections.Counter(source_input_ids_non_special_tokens)

        main_counter = main_counter + source_input_ids_counter

        # Positive example
        pos_input_ids = d[1]["input_ids"]
        pos_special_tokens_mask = d[1]['special_tokens_mask']

        pos_idx_non_special_tokens = torch.nonzero(~(pos_special_tokens_mask.type(torch.bool))).flatten()
        pos_input_ids_non_special_tokens = pos_input_ids[pos_idx_non_special_tokens]
        pos_input_ids_non_special_tokens = pos_input_ids_non_special_tokens.tolist()

        pos_input_ids_counter = collections.Counter(pos_input_ids_non_special_tokens)

        main_counter = main_counter + pos_input_ids_counter

        # Negative example
        neg_input_ids = d[2]["input_ids"]
        neg_special_tokens_mask = d[2]['special_tokens_mask']

        neg_idx_non_special_tokens = torch.nonzero(~(neg_special_tokens_mask.type(torch.bool))).flatten()
        neg_input_ids_non_special_tokens = neg_input_ids[neg_idx_non_special_tokens]
        neg_input_ids_non_special_tokens = neg_input_ids_non_special_tokens.tolist()

        neg_input_ids_counter = collections.Counter(neg_input_ids_non_special_tokens)

        main_counter = main_counter + neg_input_ids_counter

    # export total_counter
    total_count = sum(main_counter.values())

    # Divide each count with total_count
    main_counter = dict(main_counter)

    for key in main_counter.keys()
        main_counter[key] /= total_count

    # Dump to args.output
    output_file = open(args.output, 'w')
    json.dump(main_counter, output_file, indent=6)
    output_file.close()
