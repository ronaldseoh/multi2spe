import os
import argparse
import pathlib

import torch
import pytorch_lightning as pl
import ujson as json
import tqdm

from train import QuarterMaster


class Dataset:

    def __init__(self, pl_model, data_path, max_length=512, batch_size=32, ignore_num_facets=False, use_cls_for_all_facets=False):

        self.pl_model = pl_model
        self.pl_model.to('cuda')

        self.max_length = max_length
        self.batch_size = batch_size
        self.ignore_num_facets = ignore_num_facets
        self.use_cls_for_all_facets = use_cls_for_all_facets

        # data is assumed to be a json file
        with open(data_path) as f:
            # key: 'paper_id', value: paper data (including 'title', 'abstract')
            self.data = json.load(f)

        # Add extra 'unused' tokens for facets
        self.extra_facets_tokens = []

        self.batch_string_prefix = ''

        if not self.ignore_num_facets and self.pl_model.hparams.num_facets > 1:
            # If more than one facet is requested, then determine the ids
            # of "unused" tokens from the tokenizer
            # For BERT, [unused1] has the id of 1, and so on until
            # [unused99]
            for i in range(self.pl_model.hparams.num_facets - 1):
                if self.use_cls_for_all_facets:
                    self.extra_facets_tokens.append(self.pl_model.tokenizer.cls_token)
                else:
                    self.extra_facets_tokens.append('[unused{}]'.format(i+1))

            # Let tokenizer recognize our facet tokens in order to prevent it
            # from doing WordPiece stuff on these tokens
            # According to the transformers documentation, special_tokens=True prevents
            # these tokens from being normalized.
            num_added_vocabs = self.pl_model.tokenizer.add_tokens(self.extra_facets_tokens, special_tokens=True)

            if num_added_vocabs > 0:
                print("{} facet tokens were newly added to the vocabulary.".format(num_added_vocabs))

            # Crate prefixes for each batch input to be passed on to the tokenizer.
            self.batch_string_prefix += ' '.join([token for token in self.extra_facets_tokens]) + ' '

    def __len__(self):
        return len(self.data)

    def batches(self):
        # create batches
        batch = []
        batch_ids = []
        batch_size = self.batch_size

        i = 0

        for k, d in self.data.items():
            if (i) % batch_size != 0 or i == 0:
                batch_ids.append(k)

                batch.append(self.batch_string_prefix + d['title'] + ' ' + (d.get('abstract') or ''))
            else:
                input_ids = self.pl_model.tokenizer(batch, padding=True, truncation=True,
                                           return_tensors="pt", max_length=self.max_length)
                yield input_ids.to('cuda'), batch_ids

                batch_ids = [k]
                batch = [self.batch_string_prefix + d['title'] + ' ' + (d.get('abstract') or '')]

            i += 1

        if len(batch) > 0:
            input_ids = self.pl_model.tokenizer(batch, padding=True, truncation=True,
                                       return_tensors="pt", max_length=self.max_length)

            yield input_ids.to('cuda'), batch_ids


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--pl-checkpoint-path', help='path to the checkpoint saved from train.py.')
    parser.add_argument('--data-path', help='path to a json file containing paper metadata')

    parser.add_argument('--seed', default=1918, type=int)
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for prediction')
    parser.add_argument('--output', help='path to write the output embeddings file. '
                                        'the output format is jsonlines where each line has "paper_id" and "embedding" keys')

    parser.add_argument('--debug_ignore_num_facets', default=False, action='store_true')
    parser.add_argument('--debug_use_cls_for_all_facets', default=False, action='store_true')
    parser.add_argument('--debug_disable_sum_embs', default=False, action='store_true')
    parser.add_argument('--debug_disable_multiply_facet_magnitudes', default=False, action='store_true')

    args = parser.parse_args()

    # Reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    pl.seed_everything(args.seed, workers=True)

    # Load the Lightning module from the checkpoint
    model = QuarterMaster.load_from_checkpoint(args.pl_checkpoint_path, strict=False)

    # disable summed up embeddings in forward() calls if requested
    if args.debug_disable_sum_embs:
        model.sum_into_single_embeddings = "training_only"

    # disable multiplying facet magnitudes in forward() calls if requested
    if args.debug_disable_multiply_facet_magnitudes:
        model.multiply_facet_magnitudes = False

    # Put model in the eval mode
    model.eval()

    # Create a Dataset using the tokenizer and other settings in the lightning model
    dataset = Dataset(pl_model=model, data_path=args.data_path, batch_size=args.batch_size,
                      ignore_num_facets=args.debug_ignore_num_facets, use_cls_for_all_facets=args.debug_use_cls_for_all_facets)


    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    fout = open(args.output, 'a+')

    for batch, batch_ids in tqdm.auto.tqdm(dataset.batches(), total=len(dataset) // args.batch_size):

        emb = model(**batch)

        # If the model is SPECTER, then unsqueeze the second dimension in order to make emb have same dimensions as
        # quartermaster's emb
        if model.hparams.model_behavior == "specter":
            emb = emb.unsqueeze(dim=1)

        for paper_id, embedding in zip(batch_ids, emb.unbind()):
            if len(embedding.shape) == 1:
                res = {"paper_id": paper_id, "embedding": embedding.detach().cpu().numpy().tolist()}
            else:
                embedding_list = list(embedding.unbind()) # list of vectors

                for i in range(len(embedding_list)):
                    embedding_list[i] = embedding_list[i].detach().cpu().numpy().tolist()

                res = {"paper_id": paper_id, "embedding": embedding_list}

            fout.write(json.dumps(res) + '\n')

    fout.close()
