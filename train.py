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
    def __init__(self, margin=1.0, distance='l2-norm', reduction='mean'):
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

    def forward(self, query, positive, negative):
        if self.distance == 'l2-norm':
            if self.hparams.model_behavior == 'specter' or self.hparams.num_facets == 1:
                distance_positive = torch.nn.functional.pairwise_distance(query, positive)
                distance_negative = torch.nn.functional.pairwise_distance(query, negative)
            else:
                distance_positive_all = torch.cdist(query, positive, p=2).flatten(start_dim=1)
                distance_negative_all = torch.cdist(query, negative, p=2).flatten(start_dim=1)

                distance_positive = torch.min(distance_positive_all, dim=1).values
                distance_negative = torch.min(distance_negative_all, dim=1).values
                
            losses = torch.nn.functional.relu(distance_positive - distance_negative + self.margin)
        elif self.distance == 'cosine':  # independent of length
            distance_positive = torch.nn.functional.cosine_similarity(query, positive)
            distance_negative = torch.nn.functional.cosine_similarity(query, negative)
            losses = torch.nn.functional.relu(-distance_positive + distance_negative + self.margin)
        elif self.distance == 'dot':  # takes into account the length of vectors
            shapes = query.shape
            # batch dot product
            distance_positive = torch.bmm(
                query.view(shapes[0], 1, shapes[1]),
                positive.view(shapes[0], shapes[1], 1)
            ).reshape(shapes[0], )
            distance_negative = torch.bmm(
                query.view(shapes[0], 1, shapes[1]),
                negative.view(shapes[0], shapes[1], 1)
            ).reshape(shapes[0], )
            losses = torch.nn.functional.relu(-distance_positive + distance_negative + self.margin)
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

    def __init__(self, init_args):

        super().__init__()

        if isinstance(init_args, dict):
            # for loading the checkpoint, pl passes a dict (hparams are saved as dict)
            init_args = argparse.Namespace(**init_args)

        self._set_hparams(init_args)

        # NOTE: The exact model class will be transformers.BertModel
        self.model = transformers.AutoModel.from_pretrained("allenai/scibert_scivocab_cased")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
        self.tokenizer.model_max_length = self.model.config.max_position_embeddings

        self.hparams.seqlen = self.model.config.max_position_embeddings

        if self.hparams.model_behavior == 'specter':
            self.loss = TripletLoss()
        else:
            self.loss = MultiFacetTripletLoss()

        self.opt = None

    def forward(self, input_ids, token_type_ids, attention_mask):
        # in lightning, forward defines the prediction/inference actions
        source_embedding = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        if self.hparams.model_behavior == 'specter':
            return source_embedding[1]
        else:
            return source_embedding.last_hidden_state[:, 0:self.hparams.num_facets, :]

    def _get_loader(self, split):
        if split == 'train':
            fname = self.hparams.train_file
            size = self.hparams.train_size
        elif split == 'dev':
            fname = self.hparams.val_file
            size = self.hparams.val_size
        else:
            raise Exception("Invalid value for split: " + str(split))

        dataset = utils.IterableDataSetMultiWorker(file_path=fname, tokenizer=self.tokenizer, size=size)

        # pin_memory enables faster data transfer to CUDA-enabled GPU.
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
            shuffle=False, pin_memory=True)

        return loader

    def train_dataloader(self):
        return self._get_loader("train")

    def val_dataloader(self):
        return self._get_loader('dev')

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.total_gpus)  # TODO: consider num_tpu_cores

        effective_batch_size = self.hparams.batch_size * self.hparams.grad_accum * num_devices

        return (self.hparams.train_size / effective_batch_size) * self.hparams.num_epochs

    def get_lr_scheduler(self):
        get_schedule_func = ARG_TO_SCHEDULER[self.hparams.lr_scheduler]

        if self.opt is None:
            return Exception("get_lr_scheduler() should not be called before the optimizer is configured.")

        scheduler = get_schedule_func(
            self.opt,
            num_warmup_steps=int(self.hparams.warmup_frac * self.total_steps),
            num_training_steps=self.total_steps)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.hparams.adafactor:
            optimizer = transformers.optimization.Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.lr, scale_parameter=False, relative_step=False)
        else:
            optimizer = transformers.AdamW(
                optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon)

        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

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

        loss = self.loss(source_embedding, pos_embedding, neg_embedding)

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('rate', lr_scheduler.get_last_lr()[-1], on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {"loss": loss}

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

        loss = self.loss(source_embedding, pos_embedding, neg_embedding)

        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)

        return {'val_loss': loss}

    def _eval_end(self, outputs) -> tuple:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        if self.trainer.use_ddp:
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss /= self.trainer.world_size

        results = {"avg_val_loss": avg_loss}

        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.detach().cpu().item()

        return results

    def validation_epoch_end(self, outputs: list) -> dict:
        ret = self._eval_end(outputs)

        self.log('avg_val_loss', ret["avg_val_loss"], on_epoch=True, prog_bar=True)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file')
    parser.add_argument('--val_file')

    parser.add_argument('--train_size', default=684100)
    parser.add_argument('--val_size', default=145375)

    parser.add_argument('--model_behavior', default='quartermaster', choices=['quartermaster', 'specter'], type=str)
    parser.add_argument('--num_facets', default=1, type=int)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--grad_accum', default=1, type=int)
    parser.add_argument('--limit_val_batches', default=1.0, type=float) # Check 1.0 * 1 = 1 of the val set
    parser.add_argument('--val_check_interval', default=1.0, type=float) # 1.0 * 1 = Every 1 epoch
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
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

    model = QuarterMaster(args)

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

    pl_other_trainer_params = get_lightning_trainer_params(args)

    trainer = pl.Trainer(
        logger=pl_logger,
        checkpoint_callback=pl_checkpoint_callback,
        **pl_other_trainer_params)

    trainer.fit(model)
