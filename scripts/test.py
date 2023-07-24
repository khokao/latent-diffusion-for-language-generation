import shutil
from pathlib import Path

import hydra
from accelerate.logging import get_logger
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from ld4lg.cfg import MyConfig

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path='../ld4lg/cfg', config_name='config')
def main(cfg: MyConfig):
    assert cfg.infer.output_dir is not None, 'Please specify `infer.output_dir`.'

    output_dir = Path(cfg.infer.output_dir)
    cfg_path = output_dir / cfg.infer.cfg_path
    ckpt_path = output_dir / cfg.infer.ckpt_path

    loaded_cfg = OmegaConf.load(cfg_path)  # Config saved during training.
    loaded_cfg.generation = cfg.generation  # Overwrite `generation` key in saved config with hydra config.
    loaded_cfg.infer = cfg.infer  # Overwrite `infer` key in saved config with hydra config.

    module = __import__('ld4lg', fromlist=['LD4LGInterface'])
    interface_cls = getattr(module, 'LD4LGInterface')
    interface = interface_cls(loaded_cfg, output_dir, mode='test', ckpt_path=ckpt_path)

    interface.test()

    hydra_cfg = HydraConfig.get()
    created_dir = Path(hydra_cfg.run.dir)
    shutil.copy(created_dir / f'{hydra_cfg.job.name}.log', output_dir)
    shutil.rmtree(created_dir)


if __name__ == '__main__':
    main()
