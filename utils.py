import itertools

import torch

from specter.scripts.pytorch_lightning_training_script.train import (
    DataReaderFromPickled
)


class IterableDataSetMultiWorker(torch.utils.data.IterableDataset):
    def __init__(self, file_path, tokenizer, size, block_size=100, num_facets=1):
        self.datareaderfp = DataReaderFromPickled(max_sequence_length=512)
        self.data_instances = self.datareaderfp._read(file_path)
        self.tokenizer = tokenizer
        self.size = size
        self.block_size = block_size
        
        self.num_facets = num_facets
        
        self.extra_facets_tokens = []
        
        if self.num_facets > 1:
            if self.num_facets > 100:
                raise Exception("We currently only support up to 100 facets: [CLS] plus all [unused] tokens.")

            # If more than one facet is requested, then determine the ids
            # of "unused" tokens from the tokenizer
            # For BERT, [unused1] has the id of 1, and so on until
            # [unused99]
            for i in range(self.num_facets - 1):
                self.extra_facets_tokens.append('[unused{}]'.format(i+1))
                
            # Let tokenizer recognize our facet tokens in order to prevent it
            # from doing WordPiece stuff on these tokens
            # According to the transformers documentation, special_tokens=True prevents
            # these tokens from being normalized.
            num_added_vocabs = self.tokenizer.add_tokens(self.extra_facets_tokens, special_tokens=True)
            
            if num_added_vocabs > 0:
                print("{} facet tokens were newly added to the vocabulary.".format(num_added_vocabs))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_end = self.size
            for data_instance in itertools.islice(self.data_instances, iter_end):
                data_input = self.ai2_to_transformers(data_instance, self.tokenizer)
                yield data_input

        else:
            # when num_worker is greater than 1. we implement multiple process data loading.
            iter_end = self.size
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            i = 0
            for data_instance in itertools.islice(self.data_instances, iter_end):
                if int(i / self.block_size) % num_workers != worker_id:
                    i = i + 1
                    pass
                else:
                    i = i + 1
                    data_input = self.ai2_to_transformers(data_instance, self.tokenizer)
                    yield data_input

    def ai2_to_transformers(self, data_instance, tokenizer):
        """
        Args:
            data_instance: ai2 data instance
            tokenizer: huggingface transformers tokenizer
        """
        source_tokens = self.extra_facets_tokens + data_instance["source_title"].tokens

        source_title = tokenizer(' '.join([str(token) for token in source_tokens]),
                                 truncation=True, padding="max_length", return_tensors="pt",
                                 max_length=512)

        source_input = {'input_ids': source_title['input_ids'][0],
                        'token_type_ids': source_title['token_type_ids'][0],
                        'attention_mask': source_title['attention_mask'][0]}

        pos_tokens = self.extra_facets_tokens + data_instance["pos_title"].tokens

        pos_title = tokenizer(' '.join([str(token) for token in pos_tokens]),
                              truncation=True, padding="max_length", return_tensors="pt", max_length=512)

        pos_input = {'input_ids': pos_title['input_ids'][0],
                     'token_type_ids': pos_title['token_type_ids'][0],
                     'attention_mask': pos_title['attention_mask'][0]}

        neg_tokens = self.extra_facets_tokens + data_instance["neg_title"].tokens

        neg_title = tokenizer(' '.join([str(token) for token in neg_tokens]),
                              truncation=True, padding="max_length", return_tensors="pt", max_length=512)

        neg_input = {'input_ids': neg_title['input_ids'][0],
                     'token_type_ids': neg_title['token_type_ids'][0],
                     'attention_mask': neg_title['attention_mask'][0]}

        return source_input, pos_input, neg_input
