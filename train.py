import argparse
import os
import pathlib

import torch
import pytorch_lightning as pl
import transformers

# data loader classes defined in the Lightning version of specter
from specter.scripts.pytorch_lightning_training_script.train import (
    TripletLoss
)

import utils


ARG_TO_SCHEDULER = {
    "linear": transformers.optimization.get_linear_schedule_with_warmup,
    "cosine": transformers.optimization.get_cosine_schedule_with_warmup,
    "cosine_w_restarts": transformers.optimization.get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": transformers.optimization.get_polynomial_decay_schedule_with_warmup,
}

ARG_TO_SCHEDULER_CHOICES = sorted(ARG_TO_SCHEDULER.keys())
ARG_TO_SCHEDULER_METAVAR = "{" + ", ".join(ARG_TO_SCHEDULER_CHOICES) + "}"


class MultiFacetTripletLoss(torch.nn.Module):
    """
    Triplet loss function for multi-facet embeddings: Based on the TripletLoss function from  https://github.com/allenai/specter/blob/673346f9f76bcf422b38e0d1b448ef4414bcd4df/specter/model.py#L159
    """
    def __init__(self, margin=1.0, distance='l2-norm', reduction='mean', reduction_multifacet='mean'):
        """
        Args:
            margin: margin (float, optional): Default: `1`.
            distance: can be `l2-norm` or `cosine`, or `dot`
            reduction (string, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: 'mean'
        """
        super().__init__()

        self.margin = margin
        self.distance = distance
        self.reduction = reduction
        self.reduction_multifacet = reduction_multifacet

    def forward(self, query, positive, negative):
        if self.distance == 'l2-norm':
            distance_positive_all = torch.cdist(query, positive, p=2).flatten(start_dim=1)
            distance_negative_all = torch.cdist(query, negative, p=2).flatten(start_dim=1)

            if self.reduction_multifacet == 'min':
                distance_positive = torch.min(distance_positive_all, dim=1).values
                distance_negative = torch.min(distance_negative_all, dim=1).values
            elif self.reduction_multifacet == 'mean':
                distance_positive = torch.mean(distance_positive_all, dim=1)
                distance_negative = torch.mean(distance_negative_all, dim=1)

            losses = torch.nn.functional.relu(distance_positive - distance_negative + self.margin)
        else:
            raise TypeError(f"Unrecognized option for `distance`:{self.distance}")

        if self.reduction == 'mean':
            return losses.mean()

        if self.reduction == 'sum':
            return losses.sum()

        if self.reduction == 'none':
            return losses

        raise TypeError(f"Unrecognized option for `reduction`:{self.reduction}")


class QuarterMaster(pl.LightningModule):

    def __init__(self, **kwargs):

        super().__init__()

        self.save_hyperparameters()

        # NOTE: The exact model class will be transformers.BertModel
        self.model = transformers.AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

        # Extra linear layers on top of each facet embeddings
        self.extra_facet_layers = torch.nn.ModuleList()

        try:
            if self.hparams.add_extra_facet_layers:
                for _ in range(self.hparams.num_facets):
                    extra_linear = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

                    self.extra_facet_layers.append(extra_linear)

                if self.hparams.add_extra_facet_layers_initialize_0_with_nsp_weights:
                    self.extra_facet_layers[0].weight.data = self.model.pooler.dense.weight.data.clone()
                    self.extra_facet_layers[0].bias.data = self.model.pooler.dense.bias.data.clone()
        except AttributeError:
            pass

        # Extra linear layers on top of each facet embeddings
        self.extra_facet_layers_for_target = torch.nn.ModuleList()

        try:
            if self.hparams.add_extra_facet_layers_for_target:
                for _ in range(self.hparams.num_facets):
                    extra_linear = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

                    self.extra_facet_layers_for_target.append(extra_linear)

                if self.hparams.add_extra_facet_layers_initialize_0_with_nsp_weights:
                    self.extra_facet_layers_for_target[0].weight.data = self.model.pooler.dense.weight.data.clone()
                    self.extra_facet_layers_for_target[0].bias.data = self.model.pooler.dense.bias.data.clone()
        except AttributeError:
            pass

        try:
            if self.hparams.add_extra_facet_nonlinearity:
                self.extra_facet_nonlinearity = torch.nn.Tanh()
        except AttributeError:
            pass

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
        self.tokenizer.model_max_length = self.model.config.max_position_embeddings

        self.hparams.seqlen = self.model.config.max_position_embeddings

        if self.hparams.model_behavior == 'specter':
            self.loss = TripletLoss(
                margin=self.hparams.loss_margin,
                distance=self.hparams.loss_distance,
                reduction=self.hparams.loss_reduction)
        else:
            self.loss = MultiFacetTripletLoss(
                margin=self.hparams.loss_margin,
                distance=self.hparams.loss_distance,
                reduction=self.hparams.loss_reduction,
                reduction_multifacet=self.hparams.loss_reduction_multifacet)

        self.opt = None

    def forward(self, input_ids, token_type_ids, attention_mask):
        # in lightning, forward defines the prediction/inference actions
        source_embedding = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        if self.hparams.model_behavior == 'specter':
            return source_embedding[1]
        else:
            source_embedding = source_embedding.last_hidden_state

            # Extra facet layer
            # pass through the extra linear layers for each facets if enabled
            if len(self.extra_facet_layers) > 0:
                for n in range(self.hparams.num_facets):
                    source_embedding[:, n, :] = self.extra_facet_layers[n](source_embedding[:, n, :])

            return torch.nn.functional.normalize(
                source_embedding[:, 0:self.hparams.num_facets, :], p=2, dim=-1)

    def train_dataloader(self):
        dataset = torch.utils.data.BufferedShuffleDataset(
            utils.IterableDataSetMultiWorker(file_path=self.hparams.train_file, tokenizer=self.tokenizer, size=self.hparams.train_size),
            buffer_size=100)

        # pin_memory enables faster data transfer to CUDA-enabled GPU.
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        # Don't use BufferedShuffleDataset here.
        dataset = utils.IterableDataSetMultiWorker(file_path=self.hparams.val_file, tokenizer=self.tokenizer, size=self.hparams.val_size)

        # pin_memory enables faster data transfer to CUDA-enabled GPU.
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.total_gpus)  # TODO: consider num_tpu_cores

        effective_batch_size = self.hparams.batch_size * self.hparams.grad_accum * num_devices

        return int(-(-(self.hparams.train_size / effective_batch_size) * self.hparams.num_epochs // 1))

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.hparams.adafactor:
            optimizer = transformers.optimization.Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.lr, scale_parameter=False, relative_step=False)
        else:
            optimizer = transformers.AdamW(
                optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon)

        get_schedule_func = ARG_TO_SCHEDULER[self.hparams.lr_scheduler]

        scheduler = get_schedule_func(
            optimizer,
            num_warmup_steps=int(self.hparams.warmup_frac * self.total_steps),
            num_training_steps=self.total_steps)

        # Refer to https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers
        return {
            "optimizer": optimizer,
            # It seems that "frequency" needs to be 1:
            # According to the manual, "If an LR scheduler is specified for an optimizer
            # using the lr_scheduler key in the above dict,
            #"the scheduler will only be updated when its optimizer is being used."
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
        }

    def training_step(self, batch, batch_idx):

        if self.hparams.model_behavior == 'specter':
            # [1] actually contains what's referred to as "pooled output" in the Huggingface docs,
            # which is the [CLS] last hidden state followed by the BERT NSP linear layer
            source_embedding = self.model(**batch[0])[1]
            pos_embedding = self.model(**batch[1])[1]
            neg_embedding = self.model(**batch[2])[1]
        else:
            source_embedding = self.model(**batch[0]).last_hidden_state[:, 0:self.hparams.num_facets, :].contiguous()
            pos_embedding = self.model(**batch[1]).last_hidden_state[:, 0:self.hparams.num_facets, :].contiguous()
            neg_embedding = self.model(**batch[2]).last_hidden_state[:, 0:self.hparams.num_facets, :].contiguous()

            # pass through the extra linear layers for each facets if enabled
            if len(self.extra_facet_layers) > 0:
                for n in range(self.hparams.num_facets):
                    source_embedding[:, n, :] = self.extra_facet_layers[n](source_embedding[:, n, :])

            # For positive and negative papers, we could choose to train separate linear layers from source/pos
            if len(self.extra_facet_layers_for_target) > 0:
                for n in range(self.hparams.num_facets):
                    pos_embedding[:, n, :] = self.extra_facet_layers_for_target[n](pos_embedding[:, n, :])
                    neg_embedding[:, n, :] = self.extra_facet_layers_for_target[n](neg_embedding[:, n, :])
            elif len(self.extra_facet_layers) > 0:
                for n in range(self.hparams.num_facets):
                    pos_embedding[:, n, :] = self.extra_facet_layers[n](pos_embedding[:, n, :])
                    neg_embedding[:, n, :] = self.extra_facet_layers[n](neg_embedding[:, n, :])

            if self.hparams.add_extra_facet_nonlinearity:
                source_embedding = self.extra_facet_nonlinearity(source_embedding)
                pos_embedding = self.extra_facet_nonlinearity(pos_embedding)
                neg_embedding = self.extra_facet_nonlinearity(neg_embedding)

        loss = self.loss(source_embedding, pos_embedding, neg_embedding)

        self.log('train_loss', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)

        with torch.no_grad():
            # Normalize each facet embeddings for visualization purposes
            source_embedding_normalized = torch.nn.functional.normalize(source_embedding, p=2, dim=-1)
            pos_embedding_normalized = torch.nn.functional.normalize(pos_embedding, p=2, dim=-1)
            neg_embedding_normalized = torch.nn.functional.normalize(neg_embedding, p=2, dim=-1)

            source_batch_mean = torch.mean(source_embedding_normalized, dim=0, keepdims=True)
            source_loss_set_reg = torch.mean(torch.linalg.norm(source_embedding_normalized - source_batch_mean, ord=2, dim=-1))

            pos_batch_mean = torch.mean(pos_embedding_normalized, dim=0, keepdims=True)
            pos_loss_set_reg = torch.mean(torch.linalg.norm(pos_embedding_normalized - pos_batch_mean, ord=2, dim=-1))

            neg_batch_mean = torch.mean(neg_embedding_normalized, dim=0, keepdims=True)
            neg_loss_set_reg = torch.mean(torch.linalg.norm(neg_embedding_normalized - neg_batch_mean, ord=2, dim=-1))

            self.log(
                'source_loss_set_reg', source_loss_set_reg,
                on_step=True, on_epoch=False, prog_bar=False, logger=True)

            self.log(
                'pos_loss_set_reg', pos_loss_set_reg,
                on_step=True, on_epoch=False, prog_bar=False, logger=True)

            self.log(
                'neg_loss_set_reg', neg_loss_set_reg,
                on_step=True, on_epoch=False, prog_bar=False, logger=True)

            if self.hparams.num_facets > 1:
                source_facets_center_point = torch.mean(source_embedding_normalized, dim=1, keepdims=True)
                source_facets_distances_mean = torch.mean(torch.linalg.norm(source_embedding_normalized - source_facets_center_point, ord=2, dim=-1))

                pos_facets_center_point = torch.mean(pos_embedding_normalized, dim=1, keepdims=True)
                pos_facets_distances_mean = torch.mean(torch.linalg.norm(pos_embedding_normalized - pos_facets_center_point, ord=2, dim=-1))

                neg_facets_center_point = torch.mean(neg_embedding_normalized, dim=1, keepdims=True)
                neg_facets_distances_mean = torch.mean(torch.linalg.norm(neg_embedding_normalized - neg_facets_center_point, ord=2, dim=-1))

                self.log(
                    'source_facets_distances_mean', source_facets_distances_mean,
                    on_step=True, on_epoch=False, prog_bar=False, logger=True)

                self.log(
                    'pos_facets_distances_mean', pos_facets_distances_mean,
                    on_step=True, on_epoch=False, prog_bar=False, logger=True)

                self.log(
                    'neg_facets_distances_mean', neg_facets_distances_mean,
                    on_step=True, on_epoch=False, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        if self.hparams.model_behavior == 'specter':
            # [1] actually contains what's referred to as "pooled output" in the Huggingface docs,
            # which is the [CLS] last hidden state followed by the BERT NSP linear layer
            source_embedding = self.model(**batch[0])[1]
            pos_embedding = self.model(**batch[1])[1]
            neg_embedding = self.model(**batch[2])[1]
        else:
            source_embedding = self.model(**batch[0]).last_hidden_state[:, 0:self.hparams.num_facets, :]
            pos_embedding = self.model(**batch[1]).last_hidden_state[:, 0:self.hparams.num_facets, :]
            neg_embedding = self.model(**batch[2]).last_hidden_state[:, 0:self.hparams.num_facets, :]

            # pass through the extra linear layers for each facets if enabled
            if len(self.extra_facet_layers) > 0:
                for n in range(self.hparams.num_facets):
                    source_embedding[:, n, :] = self.extra_facet_layers[n](source_embedding[:, n, :])

            # For positive and negative papers, we could choose to train separate linear layers from source/pos
            if len(self.extra_facet_layers_for_target) > 0:
                for n in range(self.hparams.num_facets):
                    pos_embedding[:, n, :] = self.extra_facet_layers_for_target[n](pos_embedding[:, n, :])
                    neg_embedding[:, n, :] = self.extra_facet_layers_for_target[n](neg_embedding[:, n, :])
            elif len(self.extra_facet_layers) > 0:
                for n in range(self.hparams.num_facets):
                    pos_embedding[:, n, :] = self.extra_facet_layers[n](pos_embedding[:, n, :])
                    neg_embedding[:, n, :] = self.extra_facet_layers[n](neg_embedding[:, n, :])

            if self.hparams.add_extra_facet_nonlinearity:
                source_embedding = self.extra_facet_nonlinearity(source_embedding)
                pos_embedding = self.extra_facet_nonlinearity(pos_embedding)
                neg_embedding = self.extra_facet_nonlinearity(neg_embedding)

        loss = self.loss(source_embedding, pos_embedding, neg_embedding)

        self.log('val_loss', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)

        # Normalize each facet embeddings for visualization purposes
        source_embedding_normalized = torch.nn.functional.normalize(source_embedding, p=2, dim=-1)
        pos_embedding_normalized = torch.nn.functional.normalize(pos_embedding, p=2, dim=-1)
        neg_embedding_normalized = torch.nn.functional.normalize(neg_embedding, p=2, dim=-1)

        source_batch_mean = torch.mean(source_embedding_normalized, dim=0, keepdims=True)
        source_loss_set_reg = torch.mean(torch.linalg.norm(source_embedding_normalized - source_batch_mean, ord=2, dim=-1))

        pos_batch_mean = torch.mean(pos_embedding_normalized, dim=0, keepdims=True)
        pos_loss_set_reg = torch.mean(torch.linalg.norm(pos_embedding_normalized - pos_batch_mean, ord=2, dim=-1))

        neg_batch_mean = torch.mean(neg_embedding_normalized, dim=0, keepdims=True)
        neg_loss_set_reg = torch.mean(torch.linalg.norm(neg_embedding_normalized - neg_batch_mean, ord=2, dim=-1))

        self.log(
            'val_source_loss_set_reg', source_loss_set_reg,
            on_step=True, on_epoch=False, prog_bar=False, logger=True)

        self.log(
            'val_pos_loss_set_reg', pos_loss_set_reg,
            on_step=True, on_epoch=False, prog_bar=False, logger=True)

        self.log(
            'val_neg_loss_set_reg', neg_loss_set_reg,
            on_step=True, on_epoch=False, prog_bar=False, logger=True)

        if self.hparams.num_facets > 1:
            source_facets_center_point = torch.mean(source_embedding_normalized, dim=1, keepdims=True)
            source_facets_distances_mean = torch.mean(torch.linalg.norm(source_embedding_normalized - source_facets_center_point, ord=2, dim=-1))

            pos_facets_center_point = torch.mean(pos_embedding_normalized, dim=1, keepdims=True)
            pos_facets_distances_mean = torch.mean(torch.linalg.norm(pos_embedding_normalized - pos_facets_center_point, ord=2, dim=-1))

            neg_facets_center_point = torch.mean(neg_embedding_normalized, dim=1, keepdims=True)
            neg_facets_distances_mean = torch.mean(torch.linalg.norm(neg_embedding_normalized - neg_facets_center_point, ord=2, dim=-1))

            self.log(
                'val_source_facets_distances_mean', source_facets_distances_mean,
                on_step=True, on_epoch=False, prog_bar=False, logger=True)

            self.log(
                'val_pos_facets_distances_mean', pos_facets_distances_mean,
                on_step=True, on_epoch=False, prog_bar=False, logger=True)

            self.log(
                'val_neg_facets_distances_mean', neg_facets_distances_mean,
                on_step=True, on_epoch=False, prog_bar=False, logger=True)

        return loss

    def validation_epoch_end(self, outputs):
        # Refer to
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html
        # Note: Apparently what we get out of self.all_gather(outputs) here is a list, not a tensor
        avg_loss = torch.mean(torch.Tensor(self.all_gather(outputs)))

        if self.trainer.is_global_zero:
            self.log('avg_val_loss', avg_loss, rank_zero_only=True, on_epoch=True, prog_bar=True)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file')
    parser.add_argument('--val_file')

    parser.add_argument('--train_size', default=684100, type=int)
    parser.add_argument('--val_size', default=145375, type=int)

    parser.add_argument('--model_behavior', default='quartermaster', choices=['quartermaster', 'specter'], type=str)
    parser.add_argument('--num_facets', default=1, type=int)
    parser.add_argument('--add_extra_facet_layers', default=False, action='store_true')
    parser.add_argument('--add_extra_facet_layers_for_target', default=False, action='store_true')
    parser.add_argument('--add_extra_facet_layers_initialize_0_with_nsp_weights', default=False, action='store_true')
    parser.add_argument('--add_extra_facet_nonlinearity', default=False, action='store_true')

    parser.add_argument('--loss_margin', default=1.0, type=float)
    parser.add_argument('--loss_distance', default='l2-norm', choices=['l2-norm', 'cosine', 'dot'], type=str)
    parser.add_argument('--loss_reduction', default='mean', choices=['mean', 'sum', 'none'], type=str)
    parser.add_argument('--loss_reduction_multifacet', default='mean', choices=['mean', 'min'], type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--grad_accum', default=1, type=int)
    parser.add_argument('--limit_val_batches', default=1.0, type=float) # Check 1.0 * 1 = 1 of the val set
    parser.add_argument('--val_check_interval', default=1.0, type=float) # 1.0 * 1 = Every 1 epoch
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int, help="kwarg passed to DataLoader")
    parser.add_argument('--gpus', default='1')
    parser.add_argument('--seed', default=1918, type=int)
    parser.add_argument('--fp16', default=False, action='store_true')

    parser.add_argument("--lr_scheduler",
                        default="linear",
                        choices=ARG_TO_SCHEDULER_CHOICES,
                        metavar=ARG_TO_SCHEDULER_METAVAR,
                        type=str,
                        help="Learning rate scheduler")

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_frac", default=0.1, type=float, help="Fraction of steps to perform linear warmup.")
    parser.add_argument("--adafactor", action="store_true")

    parser.add_argument('--log_every_n_steps', default=1, type=int)
    parser.add_argument('--flush_logs_every_n_steps', default=100, type=int)
    parser.add_argument('--wandb', default=False, action="store_true")

    parser.add_argument('--save_dir', required=True)

    parsed_args = parser.parse_args()

    if ',' in parsed_args.gpus:
        parsed_args.gpus = list(map(int, parsed_args.gpus.split(',')))
        parsed_args.total_gpus = len(parsed_args.gpus)
    else:
        parsed_args.gpus = int(parsed_args.gpus)
        parsed_args.total_gpus = parsed_args.gpus

    return parsed_args


def get_lightning_trainer_params(input_args):

    train_params = {}

    train_params["accumulate_grad_batches"] = input_args.grad_accum
    train_params['track_grad_norm'] = -1
    train_params['limit_val_batches'] = input_args.limit_val_batches
    train_params['val_check_interval'] = input_args.val_check_interval
    train_params['max_epochs'] = input_args.num_epochs

    # PyTorch GPU related
    train_params["precision"] = 16 if input_args.fp16 else 32

    if (isinstance(input_args.gpus, int) and input_args.gpus > 1) or (isinstance(input_args.gpus, list) and len(input_args.gpus) > 1):
        train_params["accelerator"] = "ddp"
        # DDP optimizations
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
        train_params["plugins"] = pl.plugins.DDPPlugin(
            find_unused_parameters=False,
            gradient_as_bucket_view=True)
    else:
        train_params["accelerator"] = None

    train_params['gpus'] = input_args.gpus
    train_params['amp_backend'] = 'native' # PyTorch AMP
    train_params['deterministic'] = True
    train_params['benchmark'] = False

    # log_every_n_steps how frequently pytorch lightning logs.
    # By default, Lightning logs every 50 rows, or 50 training steps.
    train_params['log_every_n_steps'] = input_args.log_every_n_steps
    train_params['flush_logs_every_n_steps'] = input_args.flush_logs_every_n_steps

    return train_params


if __name__ == '__main__':

    args = parse_args()

    # Create args.save_dir if it doesn't exist already
    pathlib.Path(args.save_dir).mkdir(exist_ok=True)
    pathlib.Path(os.path.join(args.save_dir, 'logs')).mkdir(exist_ok=True)
    pathlib.Path(os.path.join(args.save_dir, 'checkpoints')).mkdir(exist_ok=True)

    # Reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    pl.seed_everything(args.seed, workers=True)

    model = QuarterMaster(**vars(args))

    # logger used by trainer
    if args.wandb:
        pathlib.Path(os.path.join(args.save_dir, 'logs', 'wandb')).mkdir(exist_ok=True)

        pl_logger = pl.loggers.WandbLogger(
            name=args.save_dir,
            save_dir=os.path.join(args.save_dir, 'logs'))
    else:
        pl_logger = pl.loggers.TensorBoardLogger(
            name='pl-logs',
            save_dir=os.path.join(args.save_dir, 'logs'))

    pl_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, 'checkpoints'),
        filename='ep-{epoch}_avg_val_loss-{avg_val_loss:.3f}',
        save_top_k=-1,
        save_last=1,
        every_n_val_epochs=1,
        verbose=True,
        monitor='avg_val_loss', # monitors metrics logged by self.log.
        mode='min')

    pl_learning_rate_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    pl_other_trainer_params = get_lightning_trainer_params(args)

    trainer = pl.Trainer(
        logger=pl_logger,
        callbacks=[pl_checkpoint_callback, pl_learning_rate_callback],
        **pl_other_trainer_params)

    trainer.fit(model)
