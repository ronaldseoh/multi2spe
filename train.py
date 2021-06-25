import argparse
import random
import os

import torch
import pytorch_lightning as pl
import transformers

# data loader classes defined in the Lightning version of specter
from specter.scripts.pytorch_lightning_training_script.train import (
    IterableDataSetMultiWorker,
)


ARG_TO_SCHEDULER = {
    "linear": transformers.optimization.get_linear_schedule_with_warmup,
    "cosine": transformers.optimization.get_cosine_schedule_with_warmup,
    "cosine_w_restarts": transformers.optimization.get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": transformers.optimization.get_polynomial_decay_schedule_with_warmup,
}

ARG_TO_SCHEDULER_CHOICES = sorted(ARG_TO_SCHEDULER.keys())
ARG_TO_SCHEDULER_METAVAR = "{" + ", ".join(ARG_TO_SCHEDULER_CHOICES) + "}"


class TripletLoss(torch.nn.Module):
    """
    Triplet loss: copied from  https://github.com/allenai/specter/blob/673346f9f76bcf422b38e0d1b448ef4414bcd4df/specter/model.py#L159 without any change
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
            distance_positive = torch.nn.functional.pairwise_distance(query, positive)
            distance_negative = torch.nn.functional.pairwise_distance(query, negative)
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

        self.model = transformers.AutoModel.from_pretrained("allenai/scibert_scivocab_cased")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
        self.tokenizer.model_max_length = self.model.config.max_position_embeddings

        self.hparams.seqlen = self.model.config.max_position_embeddings

        self.triple_loss = TripletLoss()

    def forward(self, input_ids, token_type_ids, attention_mask):
        # in lightning, forward defines the prediction/inference actions
        source_embedding = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return source_embedding[1]

    def _get_loader(self, split):
        if split == 'train':
            fname = self.hparams.train_file
            size = self.hparams.train_size
        elif split == 'dev':
            fname = self.hparams.dev_file
            size = self.hparams.val_size
        else:
            raise Exception("Invalid value for split: " + str(split))

        dataset = IterableDataSetMultiWorker(file_path=fname, tokenizer=self.tokenizer, size=size)

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

        return (self.hparams.training_size / effective_batch_size) * self.hparams.num_epochs

    def get_lr_scheduler(self):
        get_schedule_func = ARG_TO_SCHEDULER[self.hparams.lr_scheduler]

        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )

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
                optimizer_grouped_parameters, lr=self.hparams.lr, scale_parameter=False, relative_step=False
            )
        else:
            optimizer = transformers.AdamW(
                optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon
            )

        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        source_embedding = self.model(**batch[0])[1]

        pos_embedding = self.model(**batch[1])[1]
        neg_embedding = self.model(**batch[2])[1]

        loss = self.triple_loss(source_embedding, pos_embedding, neg_embedding)

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('rate', lr_scheduler.get_last_lr()[-1], on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        source_embedding = self.model(**batch[0])[1]

        pos_embedding = self.model(**batch[1])[1]
        neg_embedding = self.model(**batch[2])[1]

        loss = self.triple_loss(source_embedding, pos_embedding, neg_embedding)

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
    parser.add_argument('--dev_file')

    parser.add_argument('--train_size', default=684100)
    parser.add_argument('--dev_size', default=145375)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--grad_accum', default=1, type=int)
    parser.add_argument('--limit_val_batches', default=1.0, type=float)
    parser.add_argument('--val_check_interval', default=1.0, type=float)
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
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--adafactor", action="store_true")

    parser.add_argument('--log_every_n_steps', default=1, type=int)

    parser.add_argument('--save_dir', required=True)

    parsed_args = parser.parse_args()

    return parsed_args


def get_train_params(input_args):

    train_params = {}

    train_params["precision"] = 16 if input_args.fp16 else 32

    if (isinstance(input_args.gpus, int) and input_args.gpus > 1) or (isinstance(input_args.gpus, list) and len(input_args.gpus) > 1):
        train_params["distributed_backend"] = "ddp"
    else:
        train_params["distributed_backend"] = None

    train_params["accumulate_grad_batches"] = input_args.grad_accum
    train_params['track_grad_norm'] = -1
    train_params['limit_val_batches'] = input_args.limit_val_batches
    train_params['val_check_interval'] = input_args.val_check_interval
    train_params['gpus'] = input_args.gpus
    train_params['max_epochs'] = input_args.num_epochs
    
    train_params['deterministic'] = True
    train_params['benchmark'] = False

    # LOG_EVERY_N_STEPS how frequently pytorch lightning logs.
    # By default, Lightning logs every 50 rows, or 50 training steps.
    train_params['log_every_n_steps'] = input_args.log_every_n_steps

    return train_params


if __name__ == '__main__':

    args = parse_args()

    # cuBLAS reproducibility
    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    # PyTorch reproducibility
    torch.use_deterministic_algorithms(True)

    pl.seed_everything(args.seed, workers=True)

    if ',' in args.gpus:
        args.gpus = list(map(int, args.gpus.split(',')))
        args.total_gpus = len(args.gpus)
    else:
        args.gpus = int(args.gpus)
        args.total_gpus = args.gpus

    model = QuarterMaster(args)

    # default logger used by trainer
    pl_logger = pl.loggers.TensorBoardLogger(
        save_dir=args.save_dir,
        version=0,
        name='pl-logs'
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='{}/version_{}/checkpoints/'.format(args.save_dir, pl_logger.version),
        filename='ep-{epoch}_avg_val_loss-{avg_val_loss:.3f}',
        save_top_k=1,
        verbose=True,
        monitor='avg_val_loss', # monitors metrics logged by self.log.
        mode='min',
    )

    extra_train_params = get_train_params(args)

    trainer = pl.Trainer(logger=pl_logger,
                         checkpoint_callback=checkpoint_callback,
                         **extra_train_params)

    trainer.fit(model)
