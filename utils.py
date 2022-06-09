import itertools
import sys
sys.path.append(".")
sys.path.append("scincl")

import torch
# from pykeops.torch import LazyTensor
import ujson as json

from specter.scripts.pytorch_lightning_training_script.train import DataReaderFromPickled
from scincl.gdt.datasets.triples import TripleDataset


class IterableDataSetMultiWorker(torch.utils.data.IterableDataset):
    def __init__(self, file_path, tokenizer, size, block_size=100, num_facets=1, use_cls_for_all_facets=False, weights_path=None):
        # Set the options for this datareader object based on
        # the config specified in
        # https://github.com/allenai/specter/blob/master/experiment_configs/simple.jsonnet
        self.datareaderfp = DataReaderFromPickled(max_sequence_length=512, concat_title_abstract=True, lazy=True)

        self.data_instances = self.datareaderfp._read(file_path)
        self.tokenizer = tokenizer
        self.size = size
        self.block_size = block_size

        self.num_facets = num_facets
        self.use_cls_for_all_facets = use_cls_for_all_facets

        self.extra_facets_tokens = []

        if self.num_facets > 1:
            if self.num_facets > 100:
                raise Exception("We currently only support up to 100 facets: [CLS] plus all [unused] tokens.")

            # If more than one facet is requested, then determine the ids
            # of "unused" tokens from the tokenizer
            # For BERT, [unused1] has the id of 1, and so on until
            # [unused99]
            for i in range(self.num_facets - 1):
                if self.use_cls_for_all_facets:
                    self.extra_facets_tokens.append(self.tokenizer.cls_token)
                else:
                    self.extra_facets_tokens.append('[unused{}]'.format(i+1))

            # Let tokenizer recognize our facet tokens in order to prevent it
            # from doing WordPiece stuff on these tokens
            num_added_vocabs = self.tokenizer.add_special_tokens({"additional_special_tokens": self.extra_facets_tokens})

            if num_added_vocabs > 0:
                print("{} facet tokens were newly added to the vocabulary.".format(num_added_vocabs))

        self.weights = None

        if weights_path is not None:
            with open(weights_path, "r") as weights_file:
                self.weights = json.load(weights_file)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_end = self.size
            for data_instance in itertools.islice(self.data_instances, iter_end):
                data_input = self.ai2_to_transformers(data_instance)
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
                    data_input = self.ai2_to_transformers(data_instance)
                    yield data_input

    def ai2_to_transformers(self, data_instance):
        """
        Args:
            data_instance: ai2 data instance
        """
        source_tokens = self.extra_facets_tokens + data_instance["source_title"].tokens

        source_title = self.tokenizer(' '.join([str(token) for token in source_tokens]),
                                      truncation=True, padding="max_length", return_special_tokens_mask=True,
                                      return_tensors="pt", max_length=512)

        pos_tokens = self.extra_facets_tokens + data_instance["pos_title"].tokens

        pos_title = self.tokenizer(' '.join([str(token) for token in pos_tokens]),
                                   truncation=True, padding="max_length", return_special_tokens_mask=True,
                                   return_tensors="pt", max_length=512)

        neg_tokens = self.extra_facets_tokens + data_instance["neg_title"].tokens

        neg_title = self.tokenizer(' '.join([str(token) for token in neg_tokens]),
                                   truncation=True, padding="max_length", return_special_tokens_mask=True,
                                   return_tensors="pt", max_length=512)

        # As of transformers 4.9.2, adding additional special tokens do not make special_tokens_mask to make notes of them.
        # So we are masking our facet tokens manually here.
        for i in range(self.num_facets - 1):
            source_title['special_tokens_mask'][0][i+1] = 1
            pos_title['special_tokens_mask'][0][i+1] = 1
            neg_title['special_tokens_mask'][0][i+1] = 1

        source_input = {'input_ids': source_title['input_ids'][0],
                        'token_type_ids': source_title['token_type_ids'][0],
                        'attention_mask': source_title['attention_mask'][0],
                        'special_tokens_mask': source_title['special_tokens_mask'][0]}

        pos_input = {'input_ids': pos_title['input_ids'][0],
                     'token_type_ids': pos_title['token_type_ids'][0],
                     'attention_mask': pos_title['attention_mask'][0],
                     'special_tokens_mask': pos_title['special_tokens_mask'][0]}

        neg_input = {'input_ids': neg_title['input_ids'][0],
                     'token_type_ids': neg_title['token_type_ids'][0],
                     'attention_mask': neg_title['attention_mask'][0],
                     'special_tokens_mask': neg_title['special_tokens_mask'][0]}

        if self.weights is not None:
            pos_input["weights"] = torch.tensor(self.weights.get(data_instance['pos_paper_id'].metadata, 0))
            neg_input["weights"] = torch.tensor(self.weights.get(data_instance['neg_paper_id'].metadata, 0))

        return source_input, pos_input, neg_input


class SciNclTripleDataset(TripleDataset):
    def __init__(self,
                 triples_csv_path: str,
                 metadata_jsonl_path: str,
                 tokenizer,
                 sample_n: int = 0,
                 mask_anchor_tokens: bool = False,
                 predict_embeddings: bool = False,
                 abstract_only: bool = False,
                 use_cache: bool = False,
                 max_sequence_length: int = 512,
                 mlm_probability: float = 0.15,
                 graph_embeddings_path: str = None,
                 graph_paper_ids_path: str = None,
                 num_facets: int = 1,
                 use_cls_for_all_facets: bool = False):

        super().__init__(
            triples_csv_path,
            metadata_jsonl_path,
            tokenizer,
            sample_n,
            mask_anchor_tokens,
            predict_embeddings,
            abstract_only,
            use_cache,
            max_sequence_length,
            mlm_probability,
            graph_embeddings_path,
            graph_paper_ids_path,
            return_token_type_ids=True,
            return_special_tokens_mask=True)

        self.num_facets = num_facets
        self.use_cls_for_all_facets = use_cls_for_all_facets

        self.extra_facets_tokens = []

        if self.num_facets > 1:
            if self.num_facets > 100:
                raise Exception("We currently only support up to 100 facets: [CLS] plus all [unused] tokens.")

            # If more than one facet is requested, then determine the ids
            # of "unused" tokens from the tokenizer
            # For BERT, [unused1] has the id of 1, and so on until
            # [unused99]
            for i in range(self.num_facets - 1):
                if self.use_cls_for_all_facets:
                    self.extra_facets_tokens.append(self.tokenizer.cls_token)
                else:
                    self.extra_facets_tokens.append('[unused{}]'.format(i+1))

            # Let tokenizer recognize our facet tokens in order to prevent it
            # from doing WordPiece stuff on these tokens
            num_added_vocabs = self.tokenizer.add_special_tokens({"additional_special_tokens": self.extra_facets_tokens})

            if num_added_vocabs > 0:
                print("{} facet tokens were newly added to the vocabulary.".format(num_added_vocabs))

            self.paper_id_to_inputs_path += '.f{}'.format(self.num_facets)

        # Read the dataset files and load into memory
        self.load()

    def get_texts_from_ids(self, paper_ids):
        texts = super().get_texts_from_ids(paper_ids)

        facet_token_string = ' '.join([str(token) for token in self.extra_facets_tokens])

        for t in range(len(texts)):
            texts[t] = facet_token_string + texts[t]

        return texts

    def __getitem__(self, idx):
        # Put together the output from the original dataset class in a format compatible to the specter's
        output = super().__getitem__(idx)

        # As of transformers 4.9.2, adding additional special tokens do not make special_tokens_mask to make notes of them.
        # So we are masking our facet tokens manually here.
        for i in range(self.num_facets - 1):
            output['anchor_special_tokens_mask'][i+1] = 1
            output['positive_special_tokens_mask'][i+1] = 1
            output['negative_special_tokens_mask'][i+1] = 1

        source_input = {'input_ids': output['anchor_input_ids'],
                        'token_type_ids': output['anchor_token_type_ids'],
                        'attention_mask': output['anchor_attention_mask'],
                        'special_tokens_mask': output['anchor_special_tokens_mask']}

        pos_input = {'input_ids': output['positive_input_ids'],
                     'token_type_ids': output['positive_token_type_ids'],
                     'attention_mask': output['positive_attention_mask'],
                     'special_tokens_mask': output['positive_special_tokens_mask']}

        neg_input = {'input_ids': output['negative_input_ids'],
                     'token_type_ids': output['negative_token_type_ids'],
                     'attention_mask': output['negative_attention_mask'],
                     'special_tokens_mask': output['negative_special_tokens_mask']}

        return source_input, pos_input, neg_input

def batch_k_means_cosine(batch, k, n_iter=50, whitelist_masks=None):
    # Adapted the code listed in the KeOps documentation
    # https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html
    """Implements Lloyd's algorithm for the Cosine similarity metric."""

    clusters_list = []

    for b_i in range(len(batch)):
        this_example = batch[b_i]

        if whitelist_masks is not None:
            this_example = this_example[whitelist_masks[b_i].nonzero().flatten()]

        N, D = this_example.shape  # Number of samples, dimension of the ambient space

        c = this_example[:k, :].clone()  # Simplistic initialization for the centroids

        # Normalize the centroids for the cosine similarity:
        c = torch.nn.functional.normalize(c, dim=1, p=2)

        x_i = LazyTensor(this_example.view(N, 1, D))  # (N, 1, D) samples
        c_j = LazyTensor(c.view(1, k, D))  # (1, K, D) centroids

        # K-means loop:
        # - this_example is the (N, D) point cloud,
        # - cl is the (N,) vector of class labels
        # - c  is the (K, D) cloud of cluster centroids
        for i in range(n_iter):
            # E step: assign points to the closest cluster -------------------------
            S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
            cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

            # M step: update the centroids to the normalized cluster average: ------
            # Compute the sum of points per cluster:
            new_c = torch.zeros_like(c)

            # This is originally done with scatter_add_(), but that operation is not deterministic.
            # Since cl is expected to be fairly small (< 100) most of the time for our use case,
            # let's just replace this... with a loop.
            for e_i, c_i in enumerate(cl):
                new_c[c_i] += this_example[e_i]

            # Normalize the centroids, in place:
            c = torch.nn.functional.normalize(new_c, dim=1, p=2)

        clusters_list.append(c)

    clusters = torch.stack(clusters_list, dim=0)

    return clusters
