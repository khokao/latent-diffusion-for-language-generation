from dataclasses import dataclass

from hydra.conf import HydraConf


@dataclass
class DatasetConfig:
    name: str


@dataclass
class DiffusionConfig:
    beta: dict
    timesteps: int
    objective: str
    loss_type: str
    self_condition_prob: float
    class_uncondition_prob: float


@dataclass
class TransformerConfig:
    x_dim: int
    seq_len: int
    hideen_dim: int
    heads: int
    depth: int
    layer_block: tuple
    attn_dropout: float
    ff_dropout: float
    use_self_condition: bool
    use_class_condition: bool
    num_classes: int


@dataclass
class NetworkConfig:
    transformer: TransformerConfig


@dataclass
class OptimizerConfig:
    name: str
    params: dict


@dataclass
class SchedulerConfig:
    name: str
    num_warmup_steps: int


@dataclass
class ValConfig:
    interval: int
    dataloader: dict


@dataclass
class TrainConfig:
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    epoch: int
    amp: bool
    ema: bool
    wandb: bool
    log_interval: int
    save_interval: int
    dataloader: dict
    val: ValConfig


@dataclass
class GeneralConfig:
    seed: int


@dataclass
class MyConfig:
    hydra: HydraConf
    dataset: DatasetConfig
    diffusion: DiffusionConfig
    network: NetworkConfig
    train: TrainConfig
    general: GeneralConfig
