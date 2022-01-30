import argparse

import torch
import transformers

import utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file')
    parser.add_argument('--train_size', type=int)

    parser.add_argument('--pretrained_model_name', default="allenai/scibert_scivocab_uncased", type=str)

    parser.add_argument('--output', help='path to write the output embeddings file. '
                                        'the output format is jsonlines where each line has "paper_id" and "embedding" keys')

    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.pretrained_model_name)
    dataset = utils.IterableDataSetMultiWorker(file_path=args.train_file, tokenizer=tokenizer, size=args.train_size)
