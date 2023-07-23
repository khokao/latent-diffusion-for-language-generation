from pathlib import Path
import json

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from transformers import AutoTokenizer, BartForConditionalGeneration

from ld4lg.cfg import MyConfig

from .dataset import get_dataset
from .diffusion import DiffusionModel
from .models import DiffusionTransformer
from .trainer import Trainer
from .utils import freeze_model, get_length_distribution

logger = get_logger(__name__)


class LD4LGInterface:
    def __init__(self, cfg: MyConfig, output_dir, mode, ckpt_path=None):
        assert mode in ['train', 'test', 'infer']
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.mode = mode

        pretrain_name = self.cfg.network.autoencoder.name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_name)

        self._init_accelerator()
        self._init_dataset()
        self._init_distribution()
        self._init_model(pretrain_name, ckpt_path=ckpt_path)

    def _init_accelerator(self):
        project_config = ProjectConfiguration(
            project_dir=str(self.output_dir),
            # automatic_checkpoint_naming=True,
            total_limit=5,
        )
        if self.mode == 'train' and self.cfg.train.wandb:
            accelerator = Accelerator(log_with='wandb', project_config=project_config)
        else:
            accelerator = Accelerator(log_with=None, project_config=project_config)

        logger.info(f'\nAcceleratorState: {vars(accelerator.state)}')

        self.accelerator = accelerator

    def _init_dataset(self):
        logger.info(f'Initializing `{self.cfg.dataset.name}` dataset...')
        dataset = get_dataset(
            name=self.cfg.dataset.name,
            tokenizer=self.tokenizer,
            max_length=self.cfg.network.transformer.seq_len,
        )
        dataset = dataset.with_format('torch')

        split_keys = ['train', 'val', 'test']
        assert set(dataset.keys()) == set(split_keys)

        total = sum(len(dataset[k]) for k in split_keys)
        for k in split_keys:
            count = len(dataset[k])
            percentage = (count / total) * 100
            logger.info(f'{k}: {count} ({percentage:.2f}%)')

        self.train_dataset = dataset['train']
        self.val_dataset = dataset['val']
        self.test_dataset = dataset['test']

    def _init_distribution(self):
        self.length_distribution = get_length_distribution(
            dataset=self.train_dataset,
            max_seq_len=self.cfg.network.transformer.seq_len,
        )
        self.class_distribution = None  # TODO

    def _init_model(self, pretrain_name, ckpt_path=None):
        logger.info('Initializing AutoEncoder...')
        autoencoder = BartForConditionalGeneration.from_pretrained(pretrain_name)
        autoencoder = freeze_model(autoencoder)

        logger.info('Initializing Transformer...')
        transformer = DiffusionTransformer(**self.cfg.network.transformer)

        logger.info('Initializing Diffusion Model...')
        model = DiffusionModel(
            transformer=transformer,
            autoencoder=autoencoder,
            seq_len=self.cfg.network.transformer.seq_len,
            x_dim=self.cfg.network.transformer.x_dim,
            timesteps=self.cfg.diffusion.timesteps,
            beta_schedule=self.cfg.diffusion.beta.schedule,
            objective=self.cfg.diffusion.objective,
            loss_type=self.cfg.diffusion.loss_type,
            self_condition_prob=self.cfg.diffusion.self_condition_prob,
            class_uncondition_prob=self.cfg.diffusion.class_uncondition_prob,
            num_classes=self.cfg.dataset.num_classes,
            length_distribution=self.length_distribution,
            tokenizer=self.tokenizer,
        )
        if ckpt_path is not None:
            logger.info(f'Loading checkpoint from {ckpt_path}')
            model.load_state_dict(torch.load(ckpt_path))

        self.model = model



    def train(self):
        """Training.
        """
        assert self.mode == 'train'

        self.accelerator.init_trackers(
            project_name='ld4lg',
            init_kwargs={'wandb': {'name': self.output_dir.name, 'dir': self.output_dir.resolve()}},
        )

        trainer = Trainer(self.model, self.cfg, self.train_dataset, self.val_dataset, self.accelerator)
        trainer.train()

        self.accelerator.end_training()

    def test(self):
        """Testing.
        """
        raise NotImplementedError

    def infer(self):
        """Inference.
        """
        outputs = self.model.sample(
            num_samples=self.cfg.generation.num_samples,
            batch_size=self.cfg.generation.batch_size,
            class_id=self.cfg.generation.class_id,
            sampling_steps=self.cfg.generation.sampling_steps,
            strategy=self.cfg.generation.strategy,
            eta=self.cfg.generation.eta,
        )

        output_path = self.output_dir / 'infer_outputs.json'
        with output_path.open('w') as fp:
            json.dump(outputs, fp, indent=4)
