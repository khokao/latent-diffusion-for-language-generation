"""
The codes are modified.
Link:
    - [Trainer] https://github.com/Megvii-BaseDetection/YOLOX/
      blob/a5bb5ab12a61b8a25a5c3c11ae6f06397eb9b296/yolox/core/trainer.py#L36-L382
"""
from pathlib import Path
from time import time

import torch
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from ld4lg.cfg import MyConfig

from .utils import Meter

logger = get_logger(__name__)


class Trainer:
    def __init__(self, model, cfg: MyConfig, train_dataset, val_dataset, accelerator):
        self.model = model
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.accelerator = accelerator

        self.train_loader = DataLoader(self.train_dataset, **self.cfg.train.dataloader)
        self.val_loader = DataLoader(self.val_dataset, **self.cfg.train.val.dataloader)

        self.setup_training_environment()

        self.model = self.model.to(self.accelerator.device)
        if self.cfg.train.ema and self.accelerator.is_main_process:
            self.ema = EMA(
                self.model,
                beta=0.9999,
                update_after_step=100,
                update_every=10,
                inv_gamma=1.0,
                power=2 / 3,
                include_online_model=False
            )

        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler(self.optimizer)

        self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.accelerator.prepare(  # NOQA
            self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader,
        )

    def get_optimizer(self):
        # Exclude affine params in norms and bias terms.
        without_weight_decay = []
        with_weight_decay = []
        for param in self.model.parameters():
            if param.ndim < 2:
                without_weight_decay.append(param)
            else:
                with_weight_decay.append(param)
        param_groups = [
            {'params': with_weight_decay},
            {'params': without_weight_decay, 'weight_decay': 0},
        ]

        optimizer_cfg = self.cfg.train.optimizer
        optimizer_cls = getattr(torch.optim, optimizer_cfg.name)
        optimizer = optimizer_cls(param_groups, **optimizer_cfg.params)
        logger.info(f'Use {optimizer_cfg.name} optimizer')
        return optimizer

    def get_scheduler(self, optimizer):
        total_steps = self.cfg.train.epoch * self.accelerator.num_processes

        scheduler_cfg = self.cfg.train.scheduler
        scheduler = get_scheduler(
            name=scheduler_cfg.name,
            optimizer=optimizer,
            num_warmup_steps=scheduler_cfg.num_warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(f'Use `{scheduler_cfg.name}` scheduler')

        return scheduler

    def setup_training_environment(self):
        set_seed(self.cfg.general.seed)
        self.log_interval = self.cfg.train.log_interval
        self.save_interval = self.cfg.train.save_interval
        self.val_interval = self.cfg.train.val.interval
        self.ckpt_dir = Path(self.accelerator.project_dir) / 'checkpoints'
        logger.info(f'Set seed to {self.cfg["general"]["seed"]}')
        logger.info(f'Output a log for every {self.log_interval} iteration')
        logger.info(f'Save checkpoint every {self.save_interval} epoch')
        logger.info(f'Validate every {self.val_interval} epoch')
        logger.info(f'Checkpoints are saved in {self.ckpt_dir}')

    def train(self):
        self.before_train()
        self.train_in_epoch()
        self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.cfg.train.epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for batch in self.train_loader:
            self.before_iter()
            self.train_one_iter(batch)
            self.after_iter()

    def train_one_iter(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        class_id = batch['label'] if self.model.transformer.use_class_condition else None

        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, class_id=class_id)
        self.accelerator.backward(loss)

        self.accelerator.wait_for_everyone()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.train_loss_meter.update(self.accelerator.gather(loss))
        self.accelerator.wait_for_everyone()

    def before_train(self):
        self.iter = 0
        self.train_loss_meter = Meter()
        self.val_loss_meter = Meter()
        logger.info('Training start ...')

    def after_train(self):
        self.accelerator.save_state(output_dir=self.ckpt_dir)
        if self.cfg.train.ema and self.accelerator.is_main_process:
            torch.save(self.ema.ema_model.state_dict(), self.ckpt_dir / 'pytorch_model_1.bin')
            logger.info(f'EMA model states saved in {self.ckpt_dir / "pytorch_model_1.bin"}')
        logger.info('Training done')

    def before_epoch(self):
        self.model.train()
        self.epoch_start_time = time()
        logger.info(f'---> Start train epoch {self.epoch + 1}')

    def after_epoch(self):
        self.scheduler.step()

        epoch_elapsed_time = time() - self.epoch_start_time
        logger.info(f'Epoch {self.epoch + 1} done. ({epoch_elapsed_time:.1f} sec)')

        if (self.epoch + 1) % self.save_interval == 0:
            output_dir = self.ckpt_dir / f'epoch_{str(self.epoch + 1).zfill(3)}'
            self.accelerator.save_state(output_dir=output_dir)

            if self.cfg.train.ema and self.accelerator.is_main_process:
                torch.save(self.ema.ema_model.state_dict(), output_dir / 'pytorch_model_1.bin')
                logger.info(f'EMA model states saved in {output_dir / "pytorch_model_1.bin"}')

        if (self.epoch + 1) % self.val_interval == 0:
            val_model = self.ema.ema_model if self.cfg.train.ema else self.model
            self.validate(val_model=val_model, use_distributed=self.accelerator.use_distributed, prefix='val')

    def before_iter(self):
        pass

    def after_iter(self):
        if self.cfg.train.ema and self.accelerator.is_main_process:
            self.ema.update()

        if (self.iter + 1) % self.log_interval == 0:
            logger.info(
                'epoch: {}/{}, iter: {}/{}, loss: {:.3f}'.format(
                    self.epoch + 1, self.cfg.train.epoch,
                    (self.iter + 1) % len(self.train_loader), len(self.train_loader),
                    self.train_loss_meter.latest,
                )
            )
            self.accelerator.log(
                {'train_loss': self.train_loss_meter.latest},
                step=self.iter + 1,
            )
            self.accelerator.log({'train_lr': self.scheduler.get_last_lr()[0]}, step=self.iter + 1)
            self.accelerator.wait_for_everyone()
            self.train_loss_meter.reset()

        self.iter += 1

    @torch.inference_mode()
    def validate(self, val_model, use_distributed, prefix='val'):
        logger.info('Validation start...')

        val_model.eval()
        for batch in tqdm(self.val_loader, disable=not self.accelerator.is_main_process):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            class_id = batch['label'] if self.model.transformer.use_class_condition else None

            loss = self.model(input_ids=input_ids, attention_mask=attention_mask, class_id=class_id)
            loss = self.accelerator.gather(loss) if use_distributed else loss
            self.val_loss_meter.update(loss)

        logger.info(f'Validation loss (epoch: {self.epoch + 1}/{self.cfg.train.epoch}): {self.val_loss_meter.avg}')
        self.accelerator.log(
            {f'{prefix}_loss': self.val_loss_meter.avg},
            step=self.iter + 1,
        )

        if use_distributed:
            self.accelerator.wait_for_everyone()
        self.val_loss_meter.reset()
