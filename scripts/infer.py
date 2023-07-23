from pathlib import Path

import hydra
from omegaconf import OmegaConf

from ld4lg.cfg import MyConfig


@hydra.main(version_base=None, config_path='../ld4lg/cfg', config_name='config')
def main(hydra_cfg: MyConfig):
    assert hydra_cfg.infer.output_dir is not None, 'Please specify `infer.output_dir`.'

    output_dir = Path(hydra_cfg.infer.output_dir)
    cfg_path = output_dir / hydra_cfg.infer.cfg_path
    ckpt_path = output_dir / hydra_cfg.infer.ckpt_path

    loaded_cfg = OmegaConf.load(cfg_path)  # Config saved during training.
    loaded_cfg.generation = hydra_cfg.generation  # Overwrite `generation` key in saved config with hydra config.
    loaded_cfg.infere = hydra_cfg.infer  # Overwrite `infer` key in saved config with hydra config.

    module = __import__('ld4lg', fromlist=['LD4LGInterface'])
    interface_cls = getattr(module, 'LD4LGInterface')
    interface = interface_cls(loaded_cfg, output_dir, mode='infer', ckpt_path=ckpt_path)

    interface.infer()


if __name__ == '__main__':
    main()
