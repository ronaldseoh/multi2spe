import os
import itertools
import json
import typing

import torch
import transformers

from specter.scripts.pytorch_lightning_training_script.train import (
    DataReaderFromPickled
)


class IterableDataSetMultiWorker(torch.utils.data.IterableDataset):
    def __init__(self, file_path, tokenizer, size, block_size=100, num_facets=1):
        # Set the options for this datareader object based on
        # the config specified in
        # https://github.com/allenai/specter/blob/master/experiment_configs/simple.jsonnet
        self.datareaderfp = DataReaderFromPickled(max_sequence_length=512, concat_title_abstract=True, lazy=True)

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
                                 truncation=True, padding="max_length", return_tensors="pt",
                                 max_length=512)

        source_input = {'input_ids': source_title['input_ids'][0],
                        'token_type_ids': source_title['token_type_ids'][0],
                        'attention_mask': source_title['attention_mask'][0]}

        pos_tokens = self.extra_facets_tokens + data_instance["pos_title"].tokens

        pos_title = self.tokenizer(' '.join([str(token) for token in pos_tokens]),
                              truncation=True, padding="max_length", return_tensors="pt", max_length=512)

        pos_input = {'input_ids': pos_title['input_ids'][0],
                     'token_type_ids': pos_title['token_type_ids'][0],
                     'attention_mask': pos_title['attention_mask'][0]}

        neg_tokens = self.extra_facets_tokens + data_instance["neg_title"].tokens

        neg_title = self.tokenizer(' '.join([str(token) for token in neg_tokens]),
                              truncation=True, padding="max_length", return_tensors="pt", max_length=512)

        neg_input = {'input_ids': neg_title['input_ids'][0],
                     'token_type_ids': neg_title['token_type_ids'][0],
                     'attention_mask': neg_title['attention_mask'][0]}

        return source_input, pos_input, neg_input


class BertEmbeddingWithPerturbation(transformers.models.bert.modeling_bert.BertEmbeddings):
    def __init__(self, config, add_perturb_embeddings=False):
        super().__init__(config)

        self.add_perturb_embeddings = add_perturb_embeddings

        if self.add_perturb_embeddings:
            self.perturb_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

    # Ported from
    # the original BertEmbedding definition: https://github.com/huggingface/transformers/blob/41981a25cdd028007a7491d68935c8d93f9e8b47/src/transformers/models/bert/modeling_bert.py#L190
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if self.add_perturb_embeddings:
            perturb_embeddings = self.perturb_embeddings(input_ids)
            embeddings += perturb_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertLayerWithExtraLinearLayersForMultiFacets(transformers.models.bert.modeling_bert.BertLayer):
    def __init__(self, config, add_extra_facet_layers=False, num_facets=-1):
        super().__init__(config)

        self.add_extra_facet_layers = add_extra_facet_layers

        if self.add_extra_facet_layers:
            assert num_facets > 0
            self.num_facets = num_facets

            self.extra_facet_layers = torch.nn.ModuleList()

            for _ in range(self.num_facets):
                extra_linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

                self.extra_facet_layers.append(extra_linear)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # Forward through with the original implementation of forward()
        output = super().forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)

        # pass through the extra linear layers for each facets if enabled
        if len(self.extra_facet_layers) > 0:
            for n in range(self.num_facets):
                # We just need to modify output[0] == hidden state of this layer
                output[0][:, n, :] = self.extra_facet_layers[n](output[0][:, n, :])

        return output


class BertEncoderWithExtraLinearLayersForMultiFacets(transformers.models.bert.modeling_bert.BertEncoder):
    def __init__(self, config, add_extra_facet_layers_after=[], num_facets=-1):
        super().__init__(config)

        self.add_extra_facet_layers_after = add_extra_facet_layers_after

        if len(self.add_extra_facet_layers_after) > 0:
            # For layers in self.add_middle_extra_linear_after,
            # Replace the original BertLayer with the custom BertLayer class with extra linear layers
            for layer_num in self.add_extra_facet_layers_after:
                self.layer[layer_num] = BertLayerWithExtraLinearLayersForMultiFacets(
                    config, add_extra_facet_layers=True, num_facets=num_facets)


class BertModelWithExtraLinearLayersForMultiFacets(transformers.BertModel):
    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(config, add_pooling_layer)

        self.enable_extra_facets = False
        self.init_bert_layer_facet_layers = "default"

        # convert config to dict to check whether multi-facet related entries exist
        config_dict = config.to_dict()

        if 'num_facets' in kwargs:
            self.num_facets = kwargs['num_facets']
        elif 'num_facets' in config_dict.keys():
            self.num_facets = config.num_facets
        else:
            self.num_facets = 1

        if 'add_extra_facet_layers_after' in kwargs:
            self.add_extra_facet_layers_after = kwargs['add_extra_facet_layers_after']

            if len(self.add_extra_facet_layers_after) > 0:
                self.enable_extra_facets = True

                if "init_bert_layer_facet_layers" in kwargs:
                    self.init_bert_layer_facet_layers = kwargs["init_bert_layer_facet_layers"]
        else:
            if 'add_extra_facet_layers_after' in config_dict.keys():
                self.add_extra_facet_layers_after = config.add_extra_facet_layers_after

                if len(self.add_extra_facet_layers_after) > 0:
                    self.enable_extra_facets = True

                    if "init_bert_layer_facet_layers" in config_dict.keys():
                        self.init_bert_layer_facet_layers = config.init_bert_layer_facet_layers

        if self.enable_extra_facets:
            if len(self.add_extra_facet_layers_after) > 0:
                self.encoder = BertEncoderWithExtraLinearLayersForMultiFacets(config, self.add_extra_facet_layers_after, self.num_facets)

        self.add_perturb_embeddings = False

        if "add_perturb_embeddings" in kwargs:
            self.add_perturb_embeddings = kwargs["add_perturb_embeddings"]
        elif "add_perturb_embeddings" in config_dict.keys():
            self.add_perturb_embeddings = config.add_perturb_embeddings

        if self.add_perturb_embeddings:
            self.embeddings = BertEmbeddingWithPerturbation(config, add_perturb_embeddings=True)

        self.init_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: typing.Optional[typing.Union[str, os.PathLike]], *model_args, **kwargs):

        original_output_loading_info = False

        if "output_loading_info" in kwargs:
            original_output_loading_info = kwargs["output_loading_info"]

        kwargs["output_loading_info"] = True # We are forcing this to be True as we need this for the steps below

        model, loading_info = super(BertModelWithExtraLinearLayersForMultiFacets, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        if model.enable_extra_facets and not model.init_bert_layer_facet_layers == "default":
            layer_nums_without_pretrained_weights = set()

            # Initialize only if the pretrained model does not already have weights for extra_facet_layers.
            for key in loading_info["missing_keys"]:
                if key.find("extra_facet_layers") > -1 and key.endswith(".weight"):
                    layer_nums_without_pretrained_weights.add(int(key.split("extra_facet_layers.")[-1].split(".weight")[0]))

            for layer_num in layer_nums_without_pretrained_weights:
                for layer in model.encoder.layer[layer_num].extra_facet_layers:
                    if model.init_bert_layer_facet_layers == "identity":
                        torch.nn.init.eye_(layer.weight)
                        torch.nn.init.zeros_(layer.bias)

        if original_output_loading_info:
            return model, loading_info
        else:
            return model

    def save_pretrained(
        self,
        save_directory: typing.Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: typing.Optional[dict] = None,
        save_function: typing.Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        # Call the original save_pretrained() first
        super().save_pretrained(save_directory, save_config, state_dict, save_function, push_to_hub, **kwargs)

        # Edit the saved config.json
        if save_config:
            config_saved = json.load(open(os.path.join(save_directory, "config.json"), "r"))

            # Add in the entries for multi facet properties
            config_saved["num_facets"] = self.num_facets

            if self.enable_extra_facets:
                config_saved["add_extra_facet_layers_after"] = self.add_extra_facet_layers_after
                config_saved["init_bert_layer_facet_layers"] = self.init_bert_layer_facet_layers

            # Add in the entries for perturb embeddings
            if self.add_perturb_embeddings:
                config_saved["add_perturb_embeddings"] = self.add_perturb_embeddings

            # Dump the modified config back to disk
            json.dump(config_saved, open(os.path.join(save_directory, "config.json"), "w"))
